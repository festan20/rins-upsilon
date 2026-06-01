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
import os
import threading
import time

from ament_index_python.packages import get_package_share_directory
from playsound import playsound

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose, Spin
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy,
)

from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

# --------------------------------------------------------------------------
# Tuning parameters
# --------------------------------------------------------------------------
APPROACH_DISTANCE = 0.7    # metres — stop this far from a ring
LOOK_DURATION_S   = 3.0    # seconds to pause looking at a ring
NAV_TIMEOUT_S     = 30.0   # give up on a nav goal after this long
SPIN_TIMEOUT_S    = 20.0   # give up on a spin after this long
POLL_INTERVAL_S   = 0.3    # sleep between blocking polls
# Coerage waypoints [x, y, yaw_rad] in map frame.
EXPLORATION_WAYPOINTS: list[tuple[float, float, float]] = [
    (0.88, 2.0, 0.0),
    (-0.2, 2.5, 0.0),
    (-1.6, 2.0, 0.0),
    (-2.5, 1.42, 0.0)    
    
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


class Ring:
    def __init__(self, x: float, y: float, colour: str, track_id: int = -1, count: int = 1):
        self.x = x
        self.y = y
        self.colour = colour
        self.track_id = track_id
        self.count = count


class Face:
    def __init__(self, x: float, y: float, track_id: int = -1, count: int = 1):
        self.x = x
        self.y = y
        self.track_id = track_id
        self.count = count


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller')

        self._cbg = ReentrantCallbackGroup()

        # Nav2 action clients
        self._nav_client  = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                         callback_group=self._cbg)
        self._spin_client = ActionClient(self, Spin, 'spin',
                                         callback_group=self._cbg)

        # Publishers
        self._ring_markers_pub = self.create_publisher(
            MarkerArray, '/explore_ring_markers', 10)
        self._face_markers_pub = self.create_publisher(
            MarkerArray, '/explore_face_markers', 10)

        # Subscribers
        self.create_subscription(PointStamped, '/detected_rings',
                                 self._ring_cb, 10, callback_group=self._cbg)
        self.create_subscription(PointStamped, '/detected_faces',
                                 self._face_cb, 10, callback_group=self._cbg)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                 self._amcl_cb, amcl_qos, callback_group=self._cbg)

        # Robot pose
        self._current_pose_x = 0.0
        self._current_pose_y = 0.0

        # Detected rings keyed by track_id (updated with latest position/count)
        self._rings: dict[int, Ring] = {}
        # Detected faces keyed by track_id (updated with latest position/count)
        self._faces: dict[int, Face] = {}

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

    def _ring_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        # frame_id format: map/<colour>/<track_id>/<count>
        parts = msg.header.frame_id.split('/')
        colour = parts[1] if len(parts) >= 2 else 'unknown'
        track_id = int(parts[2]) if len(parts) >= 3 else -1
        count = int(parts[3]) if len(parts) >= 4 else 1

        self._rings[track_id] = Ring(x, y, colour, track_id, count)
        self.get_logger().info(
            f'Ring #{track_id} ({colour}) count={count} at ({x:.2f}, {y:.2f}) '
            f'— {len(self._rings)} unique tracks')
        self._publish_ring_markers()

    def _face_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        # frame_id format: map/<track_id>/<count>
        parts = msg.header.frame_id.split('/')
        track_id = int(parts[1]) if len(parts) >= 2 else -1
        count = int(parts[2]) if len(parts) >= 3 else 1

        self._faces[track_id] = Face(x, y, track_id, count)
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

        self.get_logger().info(
            f'Exploration done. {len(self._rings)} ring tracks detected.')

        if self._faces:
            self._visit_faces()
        else:
            self.get_logger().info('No faces found.')

        if self._rings:
            self._visit_rings()
        else:
            self.get_logger().info('No rings found.')

        self.get_logger().info('Mission complete.')

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
        for i, wp in enumerate(EXPLORATION_WAYPOINTS):
            if self._exploration_complete():
                self.get_logger().info('All targets found with sufficient detections. Stopping exploration.')
                break
            self.get_logger().info(
                f'Waypoint {i+1}/{len(EXPLORATION_WAYPOINTS)} '
                f'({wp[0]:.1f}, {wp[1]:.1f})')

            self._navigate_to(*wp)
            # self.get_logger().info('Waypoint reached, spinning 360°')
            # self._spin_360()

    # ------------------------------------------------------------------
    # Phase 2 — VISIT
    # ------------------------------------------------------------------
    def _approach_and_announce(self, x: float, y: float, label: str, sound: str) -> None:
        """Navigate to APPROACH_DISTANCE from (x, y), play sound, and pause."""
        dx = x - self._current_pose_x
        dy = y - self._current_pose_y
        dist = math.sqrt(dx * dx + dy * dy) or 0.01
        ux, uy = dx / dist, dy / dist
        ax = x - ux * APPROACH_DISTANCE
        ay = y - uy * APPROACH_DISTANCE
        yaw = math.atan2(uy, ux)

        self._navigate_to(ax, ay, yaw)

        self.get_logger().info(
            f'Arrived at {label}. Announcing for {LOOK_DURATION_S}s...')
        self._say(sound)
        time.sleep(LOOK_DURATION_S)

    def _visit_rings(self) -> None:
        sorted_rings = sorted(self._rings.values(), key=lambda r: r.count, reverse=True)
        top_rings = sorted_rings[:2]
        self.get_logger().info(
            f'Selecting top {len(top_rings)} rings by count: '
            + ', '.join(f'#{r.track_id} {r.colour} (n={r.count})' for r in top_rings))

        for i, ring in enumerate(top_rings):
            self.get_logger().info(
                f'Visiting ring {i+1}/{len(top_rings)}: '
                f'{ring.colour} (n={ring.count}) at ({ring.x:.2f}, {ring.y:.2f})')
            self._approach_and_announce(ring.x, ring.y, f'{ring.colour} ring', ring.colour)

    # ------------------------------------------------------------------
    # Phase 3 — VISIT FACES
    # ------------------------------------------------------------------
    def _visit_faces(self) -> None:
        sorted_faces = sorted(self._faces.values(), key=lambda f: f.count, reverse=True)
        top_faces = sorted_faces[:3]
        self.get_logger().info(
            f'Selecting top {len(top_faces)} faces by count: '
            + ', '.join(f'#{f.track_id} (n={f.count})' for f in top_faces))

        for i, face in enumerate(top_faces):
            self.get_logger().info(
                f'Visiting face {i+1}/{len(top_faces)}: '
                f'(n={face.count}) at ({face.x:.2f}, {face.y:.2f})')
            self._approach_and_announce(face.x, face.y, 'face', 'hello')

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------
    _SOUNDS_DIR = os.path.join(
        get_package_share_directory('upsilon'), 'sounds')

    def _say(self, colour: str) -> None:
        """Play the mp3 for a given ring colour (non-blocking)."""
        path = os.path.join(self._SOUNDS_DIR, f'{colour}.mp3')
        if not os.path.isfile(path):
            self.get_logger().warn(f'No sound file: {path}')
            return

        def _play():
            try:
                self.get_logger().info(f'Playing sound: {path}')
                playsound(path)
            except Exception as e:
                self.get_logger().warn(f'Playsound failed: {e}')
        threading.Thread(target=_play, daemon=True).start()

    # ------------------------------------------------------------------
    # Blocking navigation
    # ------------------------------------------------------------------
    def _navigate_to(self, x: float, y: float, yaw: float) -> bool:
        """Send a nav goal and block until it completes or times out."""
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
            if time.monotonic() - t0 > NAV_TIMEOUT_S:
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
            if time.monotonic() - t0 > NAV_TIMEOUT_S:
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
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
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
