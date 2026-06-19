"""Mission controller node.

Sequential flow:
    1. explore() — drives every waypoint, spins 360° at each
    2. visit()   — approaches each detected ring, pauses to look at it

Ring detections arrive via /detected_rings during exploration
(already deduplicated by ring_detector).

Runs on a MultiThreadedExecutor so that topic/action callbacks
fire while the mission thread blocks.
"""

import math
import subprocess
import threading
import time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy,
)
from std_srvs.srv import SetBool, Trigger

from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

# --------------------------------------------------------------------------
# Tuning parameters
# --------------------------------------------------------------------------
APPROACH_DISTANCE = 0.7    # metres — stop this far from a face/ring
LOOK_DURATION_S   = 5.0    # seconds to pause looking at a face/ring (also lets TTS finish)
NAV_TIMEOUT_S     = 30.0   # give up on a nav goal after this long
SPIN_TIMEOUT_S    = 20.0   # give up on a spin after this long
POLL_INTERVAL_S   = 0.3    # sleep between blocking polls
LOC_COV_THRESHOLD = 0.3    # m² (x-var + y-var) above this → localization considered bad
LOC_WAIT_TIMEOUT  = 45.0   # s — max time to wait for localization to recover
APPROACH_CANDIDATES = 24   # how many angles to try around the target
NUM_FACES_TO_VISIT = 3     # target number of faces to visit
NUM_RINGS_TO_VISIT = 2     # target number of rings to visit
# After visiting a face, navigate here before continuing to the next target.
# Set to None to skip the recovery step.
FACE_RECOVERY_WP: tuple[float, float, float] | None = (-0.46, -3.88, 0.0)
# Coverage waypoints [x, y, yaw_rad] in map frame.
EXPLORATION_WAYPOINTS: list[tuple[float, float, float]] = [
    #Task2
    (0.5, 0.2, 3.14),
    (0.5, -4.3, 0.0),
    (-1.14, -2.01, 3.14 ),
    (-2.98, -2.848, -1.57),
    (-4.29, -0.04, 0.0),
    (-2.15, 0.3284, 1.57),
    (2.7586843967437744, 0.0148270009085536, -math.pi/2), #Must be last, start of blue line

    
    # Real robot waypoints
    #(1.7515618801116943,  2.5002310276031494,  0.0),
    #(0.7185518741607666, 1.8272650241851807, 1.5),
    #(-1.4692095041275024, 2.340272617340088,   -1.5),
    #(-1.511810541152954, 1.9055171012878418,  3.0),


    # Previous sets — kept for reference
   # (0.88, 2.0, 0.0),
   #
   # (-0.2, 2.5, 0.0),
   # (-1.6, 2.0, 0.0),
    #(-2.5, 1.42, 0.0)

    #(2.0,  -0.4,  0.0),
    #(2.0,  -2.2,  0.0),
    #(0.6,  -3.7,  0.0),
    #(0.0,   -2.0, 0.0),
    #(-1.6,  -1.0, 0.0),
    #(-2.2,   0.7, 0.0),
    #(-2.3,   2.5, 0.0),
    #(0.6,    2.6, 0.0),
    #(1.0,   1.6, 0.0),
    #(-0.5,   1.2,  0.0),
    #(-1.6,  -1.0,  0.0),
]
_COLOUR_RGBA: dict[str, ColorRGBA] = {
    'red':     ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
    'blue':    ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),
    'green':   ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    'yellow':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    'orange':  ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
    'purple':  ColorRGBA(r=0.7, g=0.0, b=0.9, a=1.0),
    'black':   ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
    'unknown': ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0),
}

amcl_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class CostmapChecker:
    """Subscribes to /global_costmap/costmap and exposes is_free(x, y).

    A cell is considered free when its cost is below `threshold` (0–99) and
    is not -1 (unknown). If no costmap has been received yet, every query
    returns True so behaviour falls back to the naive approach.
    """

    def __init__(self, node: Node, topic: str = '/global_costmap/costmap'):
        self._data: tuple | None = None
        self._origin_x = 0.0
        self._origin_y = 0.0
        self._resolution = 0.05
        self._width = 0
        self._height = 0
        qos = QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        node.create_subscription(OccupancyGrid, topic, self._cb, qos)

    def _cb(self, msg: OccupancyGrid) -> None:
        self._data = msg.data
        self._origin_x = msg.info.origin.position.x
        self._origin_y = msg.info.origin.position.y
        self._resolution = msg.info.resolution
        self._width = msg.info.width
        self._height = msg.info.height

    def is_free(self, x: float, y: float, threshold: int = 50) -> bool:
        return self.cost_at(x, y) < threshold

    def cost_at(self, x: float, y: float) -> int:
        """Return costmap cost [0-254] at (x, y); 255 if unknown or out of bounds."""
        if self._data is None:
            return 0
        col = int((x - self._origin_x) / self._resolution)
        row = int((y - self._origin_y) / self._resolution)
        if not (0 <= col < self._width and 0 <= row < self._height):
            return 255
        val = self._data[row * self._width + col]
        return val if val >= 0 else 255  # -1 = unknown → treat as lethal


class Ring:
    def __init__(self, x: float, y: float, colour: str, track_id: int = -1,
                 count: int = 1, seen_order: int = 0):
        self.x = x
        self.y = y
        self.colour = colour
        self.track_id = track_id
        self.count = count
        self.seen_order = seen_order  # global discovery index (higher = seen later)


class Face:
    def __init__(self, x: float, y: float, track_id: int = -1, count: int = 1,
                 seen_order: int = 0, seen_from_x: float = 0.0, seen_from_y: float = 0.0):
        self.x = x
        self.y = y
        self.track_id = track_id
        self.seen_order = seen_order  # global discovery index (higher = seen later)
        self.count = count
        self.seen_from_x = seen_from_x  # robot position when first detected
        self.seen_from_y = seen_from_y


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller')

        self._cbg = ReentrantCallbackGroup()

        # Nav2 action clients
        self._nav_client  = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                         callback_group=self._cbg)
        self._spin_client = ActionClient(self, Spin, 'spin',
                                         callback_group=self._cbg)
        self._blue_line_client = self.create_client(SetBool, '/blue_line/set_active')

        # Publishers
        self._ring_markers_pub = self.create_publisher(
            MarkerArray, '/explore_ring_markers', 10)
        self._face_markers_pub = self.create_publisher(
            MarkerArray, '/explore_face_markers', 10)
        # TurtleBot4 onboard TTS listens on /speak
        self._speak_pub = self.create_publisher(String, '/speak', 10)
        # Latched — detectors receive current state even if they start late
        self._markers_enabled_pub = self.create_publisher(Bool, '/markers_enabled', amcl_qos)
        self._set_markers_enabled(True)
        self._report_pub = self.create_publisher(Bool, '/generate_report', 10)

        # Global costmap reader — used to pick free approach poses
        self._costmap = CostmapChecker(self)

        # Subscribers
        self.create_subscription(PointStamped, '/detected_rings_task2',
                                 self._ring_cb, 10, callback_group=self._cbg)
        self.create_subscription(PointStamped, '/detected_faces_task2',
                                 self._face_cb, 10, callback_group=self._cbg)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                 self._amcl_cb, amcl_qos, callback_group=self._cbg)
        self.create_subscription(String, '/qr_task_type',
                                 self._qr_task_cb, 10, callback_group=self._cbg)

        # Latest QR task received (set by _qr_task_cb, cleared before each face visit)
        self._qr_task: str | None = None
        self._qr_task_event = threading.Event()

        # Robot pose
        self._current_pose_x = 0.0
        self._current_pose_y = 0.0

        # Localization health (updated by _amcl_cb)
        self._localization_bad = False
        self._loc_cov = 0.0  # latest x-var + y-var from AMCL

        # Detected rings keyed by track_id (updated with latest position/count)
        self._rings: dict[int, Ring] = {}
        # Detected faces keyed by track_id (updated with latest position/count)
        self._faces: dict[int, Face] = {}
        # Global discovery counter — shared by faces and rings so we have a
        # single timeline of "what was seen when".
        self._discovery_counter = 0

        # Nav/spin async state (set by callbacks)
        self._nav_goal_handle   = None
        self._nav_result_future = None
        self._spin_goal_handle   = None
        self._spin_result_future = None

        # Start mission on a separate thread so callbacks keep firing
        self._mission_thread = threading.Thread(target=self._run_mission, daemon=True)
        self._mission_thread.start()

    # ------------------------------------------------------------------
    # Callbacks (run on executor threads)
    # ------------------------------------------------------------------
    def _amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self._current_pose_x = msg.pose.pose.position.x
        self._current_pose_y = msg.pose.pose.position.y
        cov = msg.pose.covariance
        self._loc_cov = cov[0] + cov[7]  # x-variance + y-variance
        was_bad = self._localization_bad
        self._localization_bad = self._loc_cov > LOC_COV_THRESHOLD
        if self._localization_bad and not was_bad:
            self.get_logger().warn(
                f'Localization degraded: cov={self._loc_cov:.3f} m² '
                f'(threshold {LOC_COV_THRESHOLD} m²)')
        elif not self._localization_bad and was_bad:
            self.get_logger().info(
                f'Localization recovered: cov={self._loc_cov:.3f} m²')

    def _qr_task_cb(self, msg: String) -> None:
        self._qr_task = msg.data
        self._qr_task_event.set()
        self.get_logger().info(f'QR task received: {msg.data}')

    def _ring_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        # frame_id format: map/<colour>/<track_id>/<count>
        parts = msg.header.frame_id.split('/')
        colour = parts[1] if len(parts) >= 2 else 'unknown'
        track_id = int(parts[2]) if len(parts) >= 3 else -1
        count = int(parts[3]) if len(parts) >= 4 else 1

        # Preserve discovery order for known tracks; assign next index for new ones
        if track_id in self._rings:
            seen_order = self._rings[track_id].seen_order
        else:
            seen_order = self._discovery_counter
            self._discovery_counter += 1

        self._rings[track_id] = Ring(x, y, colour, track_id, count, seen_order)
        self.get_logger().info(
            f'Ring #{track_id} ({colour}) count={count} at ({x:.2f}, {y:.2f}) '
            f'— {len(self._rings)} unique tracks')
        self._publish_ring_markers()

    def _face_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        # frame_id format: map/<track_id>/<count>/<cam_x>/<cam_y>
        parts = msg.header.frame_id.split('/')
        track_id = int(parts[1]) if len(parts) >= 2 else -1
        count = int(parts[2]) if len(parts) >= 3 else 1
        # Camera map-frame position encoded by face detector (more accurate than robot base)
        msg_cam_x = float(parts[3]) if len(parts) >= 4 else self._current_pose_x
        msg_cam_y = float(parts[4]) if len(parts) >= 5 else self._current_pose_y

        # Preserve discovery order and first-detection origin for known tracks
        if track_id in self._faces:
            seen_order = self._faces[track_id].seen_order
            seen_from_x = self._faces[track_id].seen_from_x
            seen_from_y = self._faces[track_id].seen_from_y
        else:
            seen_order = self._discovery_counter
            self._discovery_counter += 1
            seen_from_x = msg_cam_x
            seen_from_y = msg_cam_y

        self._faces[track_id] = Face(x, y, track_id, count, seen_order, seen_from_x, seen_from_y)
        self.get_logger().info(
            f'Face #{track_id} count={count} at ({x:.2f}, {y:.2f}) '
            f'— {len(self._faces)} unique tracks')
        self._publish_face_markers()

    # ------------------------------------------------------------------
    # Mission (blocking, runs on its own thread)
    # ------------------------------------------------------------------
    def _run_mission(self) -> None:
        self._wait_for_nav2()
        self.get_logger().info('Nav2 is up. Starting exploration.')

        self._explore()
        self._set_markers_enabled(False)

        self.get_logger().info('Exploration done. Visiting detected faces.')
        self._visit_faces()
        self.get_logger().info('Face visits done. Navigating to blue-line start.')


        self.get_logger().info('Starting anomaly inspection phase.')
        proc = subprocess.Popen([
            'ros2', 'run', 'upsilon', 'anomaly_controller',
            '--ros-args', '-p', 'checkpoint_set:=green',
        ])
        proc.wait()
        self.get_logger().info('Anomaly inspection done.')

        self.get_logger().info('Generating detection report.')
        self._report_pub.publish(Bool(data=True))

        last_wp = EXPLORATION_WAYPOINTS[-1]
        self._navigate_to(*last_wp, timeout=120.0)

        subprocess.run([
            'ros2', 'topic', 'pub', '--once', '/arm_command',
            'std_msgs/msg/String', "data: 'manual:[0.0, 0.6, 0.5, 2.0]'",
        ])

        if self._activate_blue_line_following():
            self.get_logger().info('Blue-line follower activated. Mission handoff complete.')
        else:
            self.get_logger().error('Blue-line follower activation failed.')

    # ------------------------------------------------------------------
    # Phase 1 — EXPLORE
    # ------------------------------------------------------------------
    def _exploration_complete(self) -> bool:
        if len(self._faces) < 3 or len(self._rings) < 2:
            return False
        top_faces = sorted(self._faces.values(), key=lambda f: f.count, reverse=True)[:3]
        top_rings = sorted(self._rings.values(), key=lambda r: r.count, reverse=True)[:2]
        return all(f.count > 15 for f in top_faces) and all(r.count > 15 for r in top_rings)

    def _explore(self) -> None:
        waypoints = EXPLORATION_WAYPOINTS[:-1]  # last waypoint is the blue-line start
        for i, wp in enumerate(waypoints):
            self.get_logger().info(
                f'Waypoint {i+1}/{len(waypoints)} '
                f'({wp[0]:.1f}, {wp[1]:.1f})')
            self._navigate_to(*wp)
            # self.get_logger().info('Waypoint reached, spinning 360°')
            # self._spin_360()

    def _visit_faces(self) -> None:
        """Approach and greet every detected face, in discovery order."""
        faces = sorted(self._faces.values(), key=lambda f: f.seen_order)
        if not faces:
            self.get_logger().warn('No faces detected — skipping face visit phase.')
            return
        for face in faces:
            self.get_logger().info(
                f'Visiting face #{face.track_id} at ({face.x:.2f}, {face.y:.2f})')
            qr_task = self._approach_and_announce(
                face.x, face.y, f'face #{face.track_id}', 'Hello',
                hint_x=face.seen_from_x, hint_y=face.seen_from_y)
            if qr_task:
                self.get_logger().info(f'Face #{face.track_id} → task: {qr_task}')
            else:
                self.get_logger().warn(f'Face #{face.track_id} → no QR task detected')
        self.get_logger().info(f'Visited {len(faces)} face(s).')

    def _activate_blue_line_following(self) -> bool:
        """Enable blue-line follower runtime mode via service call."""
        if not self._blue_line_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('/blue_line/set_active service not available.')
            return False

        req = SetBool.Request()
        req.data = True
        future = self._blue_line_client.call_async(req)

        t0 = time.monotonic()
        while not future.done():
            time.sleep(POLL_INTERVAL_S)
            if time.monotonic() - t0 > 10.0:
                self.get_logger().error('Timed out while enabling blue-line follower.')
                return False

        try:
            res = future.result()
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            self.get_logger().error(f'Failed to enable blue-line follower: {exc}')
            return False

        if not res.success:
            self.get_logger().error(f'Blue-line follower refused activation: {res.message}')
            return False

        self.get_logger().info(f'Blue-line activation response: {res.message}')
        return True

    # ------------------------------------------------------------------
    # Phase 2 — VISIT
    # ------------------------------------------------------------------
    # Shift face/ring position this far toward detection origin before sampling
    # approach candidates. Ensures the sampled point is in free space, so
    # wall-side candidates always have higher cost than open-side ones.
    _FACE_SHIFT = 0.15  # metres

    def _find_approach_pose(self, target_x: float, target_y: float,
                            hint_x: float | None = None,
                            hint_y: float | None = None,
                            ) -> tuple[float, float, float]:
        """Pick an approach pose near APPROACH_DISTANCE from the target, facing it.

        If hint_x/hint_y are given, first tries candidates within ±45° of the
        hint direction (perpendicular to the wall). Falls back to the full circle
        if none are free in that window.
        """
        # Compute preferred (perpendicular) approach angle from hint.
        if hint_x is not None and hint_y is not None:
            dx = hint_x - target_x
            dy = hint_y - target_y
            d = math.sqrt(dx * dx + dy * dy)
            if d > 0.1:
                preferred_angle = math.atan2(dy, dx)
                sample_x = target_x + self._FACE_SHIFT * dx / d
                sample_y = target_y + self._FACE_SHIFT * dy / d
                sample_dist = max(APPROACH_DISTANCE - self._FACE_SHIFT, 0.3)
            else:
                preferred_angle = None
                sample_x, sample_y = target_x, target_y
                sample_dist = APPROACH_DISTANCE
        else:
            preferred_angle = None
            sample_x, sample_y = target_x, target_y
            sample_dist = APPROACH_DISTANCE

        step = 2.0 * math.pi / APPROACH_CANDIDATES
        all_angles = [i * step for i in range(APPROACH_CANDIDATES)]

        def _angle_diff(a: float, b: float) -> float:
            d = (a - b + math.pi) % (2 * math.pi) - math.pi
            return abs(d)

        # First pass: candidates within ±45° of the preferred (perpendicular) angle.
        # Second pass: all candidates (fallback).
        window = math.radians(20)
        passes = (
            [a for a in all_angles if preferred_angle is not None
             and _angle_diff(a, preferred_angle) <= window],
            all_angles,
        ) if preferred_angle is not None else (all_angles,)

        for candidate_angles in passes:
            best_pose = None
            best_cost = 255
            for angle in candidate_angles:
                ax = sample_x + sample_dist * math.cos(angle)
                ay = sample_y + sample_dist * math.sin(angle)
                cost = self._costmap.cost_at(ax, ay)
                if cost < 50 and cost < best_cost:
                    best_cost = cost
                    best_pose = (ax, ay, math.atan2(target_y - ay, target_x - ax))
            if best_pose is not None:
                ax, ay, yaw = best_pose
                self.get_logger().info(
                    f'Approach pose at ({ax:.2f}, {ay:.2f}) cost={best_cost} '
                    f'yaw={math.degrees(yaw):+.0f}°')
                return best_pose

        # Every candidate blocked — fall back to robot-relative direction
        self.get_logger().warn(
            'No free approach pose found; falling back to robot-relative direction.')
        dx = self._current_pose_x - target_x
        dy = self._current_pose_y - target_y
        base_angle = math.atan2(dy, dx)
        ax = target_x + APPROACH_DISTANCE * math.cos(base_angle)
        ay = target_y + APPROACH_DISTANCE * math.sin(base_angle)
        return ax, ay, math.atan2(target_y - ay, target_x - ax)

    def _approach_and_announce(self, x: float, y: float, label: str, sound: str,
                               hint_x: float | None = None,
                               hint_y: float | None = None) -> str | None:
        """Navigate to APPROACH_DISTANCE from (x, y), face it, speak, and pause.

        After arriving, turns left 10° at a time until a QR task is detected
        (or 12 turns = 120° max). Returns the detected QR task string or None.
        """
        ax, ay, yaw = self._find_approach_pose(x, y, hint_x, hint_y)
        self._navigate_to(ax, ay, yaw)
        self.get_logger().info(f'Arrived at {label}. Turning left 15° then scanning for QR...')

        # Clear previous QR state, turn left 15°, wait for QR once
        self._qr_task = None
        self._qr_task_event.clear()
        self._navigate_to(ax, ay, yaw + math.radians(15))
        self._qr_task_event.wait(timeout=3.0)

        qr_task = self._qr_task
        self._say(sound)
        time.sleep(LOOK_DURATION_S)
        return qr_task

    def _visit_targets(self) -> None:
        """Visit NUM_FACES_TO_VISIT faces and NUM_RINGS_TO_VISIT rings.

        Re-scans the detection dicts before every visit so targets detected
        late (during the visit phase) still get picked up. Each iteration it
        picks the most recently discovered target (highest seen_order, a
        global index shared by faces and rings) that hasn't been visited yet.
        Stops once both quotas are met, or when no unvisited target remains.
        """
        visited_faces: set[int] = set()
        visited_rings: set[int] = set()

        while True:
            need_faces = len(visited_faces) < NUM_FACES_TO_VISIT
            need_rings = len(visited_rings) < NUM_RINGS_TO_VISIT
            if not need_faces and not need_rings:
                break  # both quotas met

            # Re-scan: collect unvisited targets of the kinds still needed
            candidates: list[tuple[str, object]] = []
            if need_faces:
                candidates += [('face', f) for f in self._faces.values()
                               if f.track_id not in visited_faces]
            if need_rings:
                candidates += [('ring', r) for r in self._rings.values()
                               if r.track_id not in visited_rings]

            if not candidates:
                self.get_logger().warn(
                    f'No more targets to visit — visited '
                    f'{len(visited_faces)}/{NUM_FACES_TO_VISIT} faces, '
                    f'{len(visited_rings)}/{NUM_RINGS_TO_VISIT} rings.')
                break

            # Most recently discovered first (global order across faces+rings)
            kind, target = max(candidates, key=lambda kv: kv[1].seen_order)

            if kind == 'face':
                label, phrase = 'face', 'Hello'
                visited_faces.add(target.track_id)
            else:
                label = f'{target.colour} ring'
                phrase = f'{target.colour} ring'
                visited_rings.add(target.track_id)

            self.get_logger().info(
                f'Visiting {label} #{target.track_id} (n={target.count}) '
                f'at ({target.x:.2f}, {target.y:.2f}) — '
                f'{len(visited_faces)}/{NUM_FACES_TO_VISIT} faces, '
                f'{len(visited_rings)}/{NUM_RINGS_TO_VISIT} rings')
            self._approach_and_announce(target.x, target.y, label, phrase)

    # ------------------------------------------------------------------
    # Marker enable/disable
    # ------------------------------------------------------------------
    def _set_markers_enabled(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled
        self._markers_enabled_pub.publish(msg)
        self.get_logger().info(f'Detector markers {"enabled" if enabled else "disabled"}.')

    # ------------------------------------------------------------------
    # Speech (TurtleBot4 onboard TTS listens on /speak)
    # ------------------------------------------------------------------
    def _say(self, phrase: str) -> None:
        """Publish a phrase for the robot's onboard TTS to speak."""
        msg = String()
        msg.data = phrase
        self._speak_pub.publish(msg)
        self.get_logger().info(f'Speaking: "{phrase}"')

    # ------------------------------------------------------------------
    # Localization recovery
    # ------------------------------------------------------------------
    def _relocalize(self) -> None:
        """Spin in place until AMCL covariance recovers or we time out."""
        self.get_logger().warn(
            f'Relocalizing (cov={self._loc_cov:.3f} m²) — spinning to gather scan data...')
        t0 = time.monotonic()
        while self._localization_bad:
            if time.monotonic() - t0 > LOC_WAIT_TIMEOUT:
                self.get_logger().error(
                    f'Relocalization timed out after {LOC_WAIT_TIMEOUT:.0f}s '
                    f'(cov={self._loc_cov:.3f} m²) — proceeding anyway.')
                break
            self._spin_360()
            time.sleep(0.5)
        if not self._localization_bad:
            self.get_logger().info(
                f'Relocalization succeeded (cov={self._loc_cov:.3f} m²).')

    # ------------------------------------------------------------------
    # Blocking navigation
    # ------------------------------------------------------------------
    def _navigate_to(self, x: float, y: float, yaw: float,
                     timeout: float = NAV_TIMEOUT_S) -> bool:
        """Send a nav goal and block until it completes or times out."""
        if self._localization_bad:
            self._relocalize()

        pose = self._make_pose(x, y, yaw)

        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 not available.')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        send_future = self._nav_client.send_goal_async(goal_msg)

        # Wait for goal acceptance
        t0 = time.monotonic()
        while not send_future.done():
            time.sleep(POLL_INTERVAL_S)
            if time.monotonic() - t0 > timeout:
                self.get_logger().warn('Nav goal acceptance timed out.')
                return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Nav goal rejected.')
            return False

        # Wait for result
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(POLL_INTERVAL_S)
            if self._localization_bad:
                self.get_logger().warn('Localization degraded mid-nav — cancelling goal to relocalize.')
                goal_handle.cancel_goal_async()
                time.sleep(1.0)  # let cancel propagate
                self._relocalize()
                return self._navigate_to(x, y, yaw, timeout)
            if time.monotonic() - t0 > timeout:
                self.get_logger().warn('Nav goal timed out; cancelling.')
                goal_handle.cancel_goal_async()
                return False

        return True

    def _spin_360(self) -> bool:
        """Send a 360° spin and block until done or timed out."""
        if not self._spin_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Spin action not available; skipping.')
            return False

        goal_msg = Spin.Goal()
        goal_msg.target_yaw = 2.0 * math.pi
        send_future = self._spin_client.send_goal_async(goal_msg)

        t0 = time.monotonic()
        while not send_future.done():
            time.sleep(POLL_INTERVAL_S)
            if time.monotonic() - t0 > SPIN_TIMEOUT_S:
                self.get_logger().warn('Spin acceptance timed out.')
                return False

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Spin goal rejected.')
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(POLL_INTERVAL_S)
            if time.monotonic() - t0 > SPIN_TIMEOUT_S:
                self.get_logger().warn('Spin timed out; cancelling.')
                goal_handle.cancel_goal_async()
                return False

        return True

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def _make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp    = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation = self._yaw_to_quat(yaw)
        return ps

    @staticmethod
    def _yaw_to_quat(yaw: float) -> Quaternion:
        q = quaternion_from_euler(0.0, 0.0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    # ------------------------------------------------------------------
    # Ring markers
    # ------------------------------------------------------------------
    def _publish_ring_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for ring in self._rings.values():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns     = 'detected_rings'
            m.id     = ring.track_id
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = ring.x
            m.pose.position.y = ring.y
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color = _COLOUR_RGBA.get(ring.colour, _COLOUR_RGBA['unknown'])
            arr.markers.append(m)
        self._ring_markers_pub.publish(arr)

    # ------------------------------------------------------------------
    # Face markers
    # ------------------------------------------------------------------
    def _publish_face_markers(self) -> None:
        now = self.get_clock().now().to_msg()
        arr = MarkerArray()
        for face in self._faces.values():
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns     = 'detected_faces'
            m.id     = face.track_id
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = face.x
            m.pose.position.y = face.y
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            arr.markers.append(m)
        self._face_markers_pub.publish(arr)

    # ------------------------------------------------------------------
    # Nav2 readiness
    # ------------------------------------------------------------------
    def _wait_for_nav2(self) -> None:
        from lifecycle_msgs.srv import GetState
        self.get_logger().info('Waiting for Nav2...')
        for node_name in ('bt_navigator', 'amcl'):
            svc    = f'{node_name}/get_state'
            client = self.create_client(GetState, svc, callback_group=self._cbg)
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {svc}...')
            req   = GetState.Request()
            state = ''
            while state != 'active':
                future = client.call_async(req)
                # Poll instead of spin_until_future_complete (we're on a non-executor thread)
                while not future.done():
                    time.sleep(0.2)
                if future.result():
                    state = future.result().current_state.label
                time.sleep(1.0)
            self.get_logger().info(f'{node_name} is active.')


def main():
    rclpy.init()
    node = ControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
