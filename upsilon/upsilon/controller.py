"""Mission controller node.

FSM (3-phase):
    EXPLORE  — drives every waypoint in order; spins 360° on arrival;
               buffers all face/ring detections and shows live RViz markers
    MATCH    — clusters raw detections spatially; majority-votes ring colours
    VISIT    — approaches each matched target, speaks, then moves to the next
    DONE     — all matched targets visited; announces mission complete

Detection callbacks during EXPLORE only buffer data — no interruptions.

Run with a MultiThreadedExecutor so that action callbacks and topic
callbacks can execute concurrently.
"""

import math
import time
from enum import Enum, auto

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from geometry_msgs.msg import PointStamped
from nav2_msgs.action import NavigateToPose, Spin
from std_msgs.msg import ColorRGBA, String
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

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
APPROACH_DISTANCE = 0.7    # metres — stop this far from a target
INTERACT_WAIT_S   = 3.0    # seconds to wait after speaking
WAYPOINT_TIMEOUT_S = 30.0  # give up on a waypoint nav after this long
CLUSTER_RADIUS     = 1.0   # metres — detections within this radius → same target

# Coverage waypoints [x, y, yaw_rad] in map frame.
EXPLORATION_WAYPOINTS: list[tuple[float, float, float]] = [
    (2.0,  -0.4,  0.0),
    (2.0,  -2.2,  0.0),
    (0.6,  -3.7,  0.0),
    (0.0,   -2.0, 0.0),
    (-1.6,  -1.0, 0.0),
    (-2.2,   0.7, 0.0),
    (-2.3,   2.5, 0.0),
    (0.6,    2.6, 0.0),
    (1.0,   1.6, 0.0),
    (-0.5,   1.2,  0.0),
    (-1.6,  -1.0,  0.0),
]

# Colour → RGBA (for explore markers)
_COLOUR_RGBA: dict[str, ColorRGBA] = {
    'blue':    ColorRGBA(r=0.0, g=0.3, b=1.0, a=1.0),
    'green':   ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
    'yellow':  ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
    'orange':  ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0),
    'purple':  ColorRGBA(r=0.7, g=0.0, b=0.9, a=1.0),
    'black':   ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
    'unknown': ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0),
}

# ---------------------------------------------------------------------------
amcl_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class State(Enum):
    EXPLORE = auto()
    MATCH   = auto()
    VISIT   = auto()
    DONE    = auto()


class Target:
    def __init__(self, kind: str, x: float, y: float, colour: str = ''):
        self.kind   = kind      # 'face' or 'ring'
        self.x      = x
        self.y      = y
        self.colour = colour    # ring colour or ''


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
        self._speech_pub = self.create_publisher(String, '/speech', 10)
        self._explore_face_markers_pub = self.create_publisher(
            MarkerArray, '/explore_face_markers', 10)
        self._explore_ring_markers_pub = self.create_publisher(
            MarkerArray, '/explore_ring_markers', 10)

        # Subscribers
        self.create_subscription(PointStamped, '/detected_faces',
                                 self._face_cb, 10, callback_group=self._cbg)
        self.create_subscription(PointStamped, '/detected_rings',
                                 self._ring_cb, 10, callback_group=self._cbg)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                 self._amcl_cb, amcl_qos, callback_group=self._cbg)

        # --- Robot pose ---
        self._current_pose_x = 0.0
        self._current_pose_y = 0.0

        # --- FSM ---
        self._state = State.EXPLORE

        # --- EXPLORE sub-state ---
        self._explore_sub = 'drive'   # 'drive' | 'spin'
        self._waypoint_idx = 0

        # --- Raw detection buffers ---
        self._raw_faces: list[tuple[float, float]]        = []
        self._raw_rings: list[tuple[float, float, str]]   = []

        # --- Nav state ---
        self._nav_goal_handle  = None
        self._nav_result_future = None
        self._nav_sent_time: float | None = None

        # --- Spin state ---
        self._spin_goal_handle   = None
        self._spin_result_future = None

        # --- VISIT state ---
        self._targets_to_visit: list[Target] = []
        self._visit_idx   = 0
        self._visit_sub   = 'drive'   # 'drive' | 'interact'
        self._visit_target: Target | None  = None
        self._visit_interact_start: float | None = None

        # Main FSM timer
        self._timer = self.create_timer(0.5, self._fsm_tick, callback_group=self._cbg)

        self.get_logger().info('Controller ready. Waiting for Nav2...')
        self._wait_for_nav2()
        self.get_logger().info('Nav2 is up. Starting exploration.')

        # Kick off first waypoint
        self._advance_waypoint()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self._current_pose_x = msg.pose.pose.position.x
        self._current_pose_y = msg.pose.pose.position.y

    def _face_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        self._raw_faces.append((x, y))
        self.get_logger().info(f'Raw face buffered at ({x:.2f}, {y:.2f})')
        self._publish_explore_markers()

    def _ring_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        colour = msg.header.frame_id.split('/')[-1] if '/' in msg.header.frame_id else 'unknown'
        self._raw_rings.append((x, y, colour))
        self.get_logger().info(f'Raw ring ({colour}) buffered at ({x:.2f}, {y:.2f})')
        self._publish_explore_markers()

    # ------------------------------------------------------------------
    # FSM dispatcher
    # ------------------------------------------------------------------
    def _fsm_tick(self) -> None:
        if   self._state == State.EXPLORE: self._tick_explore()
        elif self._state == State.MATCH:   self._tick_match()
        elif self._state == State.VISIT:   self._tick_visit()
        elif self._state == State.DONE:    pass

    # ------------------------------------------------------------------
    # Phase 1 — EXPLORE
    # ------------------------------------------------------------------
    def _tick_explore(self) -> None:
        if self._explore_sub == 'drive':
            if self._nav_is_complete():
                self._explore_sub = 'spin'
                self.get_logger().info(
                    f'Waypoint reached, spinning 360°')
                self._send_spin_goal(2.0 * math.pi)

        elif self._explore_sub == 'spin':
            if self._spin_is_complete():
                if self._waypoint_idx >= len(EXPLORATION_WAYPOINTS):
                    self.get_logger().info(
                        'All waypoints explored. Transitioning to MATCH.')
                    self._state = State.MATCH
                    return
                self._advance_waypoint()
                self._explore_sub = 'drive'

    # ------------------------------------------------------------------
    # Phase 2 — MATCH (synchronous, runs once)
    # ------------------------------------------------------------------
    def _tick_match(self) -> None:
        self._targets_to_visit = self._run_match()
        self._visit_idx = 0
        self.get_logger().info(
            f'MATCH produced {len(self._targets_to_visit)} targets to visit.')
        self._state = State.VISIT if self._targets_to_visit else State.DONE
        if self._state == State.DONE:
            self._speak('Mission complete. No targets found.')

    def _run_match(self) -> list[Target]:
        targets: list[Target] = []

        # --- Face clusters ---
        face_pts = list(self._raw_faces)
        for idx_group in self._cluster_indexed(face_pts, CLUSTER_RADIUS):
            pts = [face_pts[i] for i in idx_group]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            targets.append(Target('face', cx, cy))
            self.get_logger().info(
                f'Face cluster of {len(pts)} → ({cx:.2f}, {cy:.2f})')

        # --- Ring clusters ---
        ring_pts = [(x, y) for x, y, _ in self._raw_rings]
        for idx_group in self._cluster_indexed(ring_pts, CLUSTER_RADIUS):
            pts     = [ring_pts[i]          for i in idx_group]
            colours = [self._raw_rings[i][2] for i in idx_group]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            colour = self._majority_vote(colours)
            targets.append(Target('ring', cx, cy, colour))
            self.get_logger().info(
                f'Ring cluster of {len(pts)} → ({cx:.2f}, {cy:.2f}) colour={colour}')

        return targets

    @staticmethod
    def _cluster_indexed(
            points: list[tuple[float, float]],
            radius: float,
    ) -> list[list[int]]:
        """Greedy single-linkage BFS clustering. Returns lists of indices."""
        assigned = [-1] * len(points)
        clusters: list[list[int]] = []
        for i in range(len(points)):
            if assigned[i] != -1:
                continue
            cid = len(clusters)
            clusters.append([i])
            assigned[i] = cid
            frontier = [i]
            while frontier:
                fi = frontier.pop()
                xf, yf = points[fi]
                for j, (xj, yj) in enumerate(points):
                    if assigned[j] != -1:
                        continue
                    if math.sqrt((xf - xj) ** 2 + (yf - yj) ** 2) < radius:
                        assigned[j] = cid
                        clusters[cid].append(j)
                        frontier.append(j)
        return clusters

    @staticmethod
    def _majority_vote(colours: list[str]) -> str:
        counts: dict[str, int] = {}
        for c in colours:
            counts[c] = counts.get(c, 0) + 1
        return max(counts, key=lambda k: counts[k])

    # ------------------------------------------------------------------
    # Phase 3 — VISIT
    # ------------------------------------------------------------------
    def _tick_visit(self) -> None:
        if self._visit_idx >= len(self._targets_to_visit):
            self._state = State.DONE
            self._speak('Mission complete. All targets visited.')
            self.get_logger().info('DONE — all targets visited.')
            return

        if self._visit_sub == 'drive':
            t = self._targets_to_visit[self._visit_idx]
            # Lazy-start: only send goal when we switch to a new target
            if self._visit_target is not t:
                self._visit_target = t
                self.get_logger().info(
                    f'Approaching target {self._visit_idx}: '
                    f'{t.kind} at ({t.x:.2f}, {t.y:.2f})')
                self._send_approach_goal(t)

            if self._nav_is_complete():
                self._visit_sub = 'interact'
                self._visit_interact_start = time.monotonic()
                self._speak_to_target(t)

        elif self._visit_sub == 'interact':
            if time.monotonic() - self._visit_interact_start < INTERACT_WAIT_S:
                return
            self.get_logger().info(
                f'Finished target {self._visit_idx}. '
                f'Moving to next.')
            self._visit_idx += 1
            self._visit_target = None
            self._visit_sub = 'drive'

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def _send_approach_goal(self, target: Target) -> None:
        self._send_nav_goal(self._make_approach_pose(target))

    def _advance_waypoint(self) -> None:
        wp = EXPLORATION_WAYPOINTS[self._waypoint_idx % len(EXPLORATION_WAYPOINTS)]
        self._waypoint_idx += 1
        self.get_logger().info(
            f'Navigating to waypoint {self._waypoint_idx} ({wp[0]:.1f}, {wp[1]:.1f})')
        self._send_nav_goal(self._make_pose(*wp))

    def _send_nav_goal(self, pose: PoseStamped) -> None:
        self._nav_result_future = None
        self._nav_goal_handle   = None
        self._nav_sent_time     = time.monotonic()

        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Nav2 not available.')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._nav_goal_response_cb)

    def _nav_goal_response_cb(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Nav goal rejected.')
            return
        self._nav_goal_handle   = handle
        self._nav_result_future = handle.get_result_async()

    def _nav_is_complete(self) -> bool:
        if self._nav_result_future is None:
            return True
        if not self._nav_result_future.done():
            if (self._nav_sent_time and
                    time.monotonic() - self._nav_sent_time > WAYPOINT_TIMEOUT_S):
                self.get_logger().warn('Waypoint timed out; skipping.')
                self._cancel_nav()
                return True
            return False
        self._nav_result_future = None
        return True

    def _cancel_nav(self) -> None:
        if self._nav_goal_handle is not None:
            self._nav_goal_handle.cancel_goal_async()
            self._nav_goal_handle = None
        self._nav_result_future = None

    # ------------------------------------------------------------------
    # Spin helpers
    # ------------------------------------------------------------------
    def _send_spin_goal(self, yaw: float) -> None:
        self._spin_result_future = None
        self._spin_goal_handle   = None

        if not self._spin_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Spin action server not available; skipping spin.')
            return

        goal_msg = Spin.Goal()
        goal_msg.target_yaw = yaw
        future = self._spin_client.send_goal_async(goal_msg)
        future.add_done_callback(self._spin_goal_response_cb)

    def _spin_goal_response_cb(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Spin goal rejected.')
            return
        self._spin_goal_handle   = handle
        self._spin_result_future = handle.get_result_async()

    def _spin_is_complete(self) -> bool:
        if self._spin_result_future is None:
            return True
        if not self._spin_result_future.done():
            return False
        self._spin_result_future = None
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

    def _make_approach_pose(self, target: Target) -> PoseStamped:
        dx   = target.x - self._current_pose_x
        dy   = target.y - self._current_pose_y
        dist = math.sqrt(dx * dx + dy * dy) or 0.01
        ux, uy = dx / dist, dy / dist
        ax = target.x - ux * APPROACH_DISTANCE
        ay = target.y - uy * APPROACH_DISTANCE
        return self._make_pose(ax, ay, math.atan2(uy, ux))

    @staticmethod
    def _yaw_to_quat(yaw: float) -> Quaternion:
        q = quaternion_from_euler(0.0, 0.0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    # ------------------------------------------------------------------
    # Speech
    # ------------------------------------------------------------------
    def _speak_to_target(self, target: Target) -> None:
        if target.kind == 'face':
            self._speak('Hello!')
        else:
            self._speak(f'I found a {target.colour or "unknown"} ring.')

    def _speak(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._speech_pub.publish(msg)

    # ------------------------------------------------------------------
    # Exploration markers
    # ------------------------------------------------------------------
    def _publish_explore_markers(self) -> None:
        now = self.get_clock().now().to_msg()

        # --- Face markers (yellow spheres) ---
        face_arr = MarkerArray()
        for i, (x, y) in enumerate(self._raw_faces):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns     = 'raw_faces'
            m.id     = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 1.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
            face_arr.markers.append(m)
        self._explore_face_markers_pub.publish(face_arr)

        # --- Ring markers (colour-coded spheres) ---
        ring_arr = MarkerArray()
        for i, (x, y, colour) in enumerate(self._raw_rings):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns     = 'raw_rings'
            m.id     = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = x
            m.pose.position.y = y
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color = _COLOUR_RGBA.get(colour, _COLOUR_RGBA['unknown'])
            ring_arr.markers.append(m)
        self._explore_ring_markers_pub.publish(ring_arr)

    # ------------------------------------------------------------------
    # Nav2 readiness
    # ------------------------------------------------------------------
    def _wait_for_nav2(self) -> None:
        from lifecycle_msgs.srv import GetState
        for node_name in ('bt_navigator', 'amcl'):
            svc    = f'{node_name}/get_state'
            client = self.create_client(GetState, svc, callback_group=self._cbg)
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {svc}...')
            req   = GetState.Request()
            state = ''
            while state != 'active':
                future = client.call_async(req)
                rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
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
