"""Mission controller node.

FSM:
    EXPLORE        — cycles through hard-coded coverage waypoints via Nav2
    APPROACH       — navigates to a point ~0.7 m in front of a detected target
    INTERACT       — speaks to the target, waits, marks as visited
    DONE           — all 3 faces + 2 rings found; announces mission complete

Detection callbacks push new targets onto a priority queue; when the robot is
in EXPLORE mode the highest-priority pending target is addressed first.

Run with a MultiThreadedExecutor so that action callbacks and topic callbacks
can execute concurrently:

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
"""

import math
import time
from enum import Enum, auto

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Quaternion
from nav2_msgs.action import NavigateToPose, Spin
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy,
    qos_profile_sensor_data,
)

from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
TOTAL_FACES = 3
TOTAL_RINGS = 2
APPROACH_DISTANCE = 0.7   # metres — stop this far from a target
INTERACT_WAIT_S = 3.0     # seconds to wait after speaking
WAYPOINT_TIMEOUT_S = 30.0  # give up on a waypoint after this long

# Coverage waypoints [x, y, yaw_rad] in map frame.
# These are placeholder values — tune to each competition world.
EXPLORATION_WAYPOINTS: list[tuple[float, float, float]] = [
    (1.0,  -3.5,  0.0),
    (-1.0,  -2.5,  0.0)
]

# ---------------------------------------------------------------------------
amcl_qos = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class State(Enum):
    EXPLORE = auto()
    APPROACH = auto()
    INTERACT = auto()
    DONE = auto()


class Target:
    def __init__(self, kind: str, x: float, y: float, colour: str = ''):
        self.kind = kind      # 'face' or 'ring'
        self.x = x
        self.y = y
        self.colour = colour  # ring colour or ''


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller')

        # Use a reentrant callback group so action clients and timers coexist
        self._cbg = ReentrantCallbackGroup()

        # Nav2 action clients
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                        callback_group=self._cbg)
        self._spin_client = ActionClient(self, Spin, 'spin',
                                         callback_group=self._cbg)

        # Publishers
        self._speech_pub = self.create_publisher(String, '/speech', 10)

        # Subscribers
        self.create_subscription(PointStamped, '/detected_faces', self._face_cb, 10,
                                 callback_group=self._cbg)
        self.create_subscription(PointStamped, '/detected_rings', self._ring_cb, 10,
                                 callback_group=self._cbg)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose',
                                 self._amcl_cb, amcl_qos, callback_group=self._cbg)

        # State
        self._state = State.EXPLORE
        self._current_pose_x = 0.0
        self._current_pose_y = 0.0
        self._waypoint_idx = 0
        self._pending_targets: list[Target] = []
        self._current_target: Target | None = None
        self._faces_found = 0
        self._rings_found = 0
        self._interact_start: float | None = None
        self._nav_goal_handle = None
        self._nav_result_future = None
        self._nav_sent_time: float | None = None

        # Main FSM timer
        self._timer = self.create_timer(0.5, self._fsm_tick, callback_group=self._cbg)

        self.get_logger().info('Controller ready. Waiting for Nav2...')
        self._wait_for_nav2()
        self.get_logger().info('Nav2 is up. Starting exploration.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self._current_pose_x = msg.pose.pose.position.x
        self._current_pose_y = msg.pose.pose.position.y

    def _face_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        if self._already_visited('face', x, y):
            return
        self.get_logger().info(f'Face queued at ({x:.2f}, {y:.2f})')
        self._pending_targets.append(Target('face', x, y))

    def _ring_cb(self, msg: PointStamped) -> None:
        x, y = msg.point.x, msg.point.y
        # Colour is encoded in frame_id as "map/<colour>"
        colour = 'unknown'
        if '/' in msg.header.frame_id:
            colour = msg.header.frame_id.split('/')[-1]
        if self._already_visited('ring', x, y):
            return
        self.get_logger().info(f'Ring ({colour}) queued at ({x:.2f}, {y:.2f})')
        self._pending_targets.append(Target('ring', x, y, colour))

    # ------------------------------------------------------------------
    # FSM
    # ------------------------------------------------------------------
    def _fsm_tick(self) -> None:
        if self._state == State.EXPLORE:
            self._tick_explore()
        elif self._state == State.APPROACH:
            self._tick_approach()
        elif self._state == State.INTERACT:
            self._tick_interact()
        elif self._state == State.DONE:
            pass  # nothing to do

    def _tick_explore(self) -> None:
        # If a new target appeared, switch to approach
        if self._pending_targets:
            target = self._pending_targets.pop(0)
            self._current_target = target
            self._cancel_nav()
            self._state = State.APPROACH
            self.get_logger().info(
                f'Interrupting exploration for {target.kind} at ({target.x:.2f}, {target.y:.2f})'
            )
            self._send_approach_goal(target)
            return

        # Continue exploration: check if current waypoint nav is done
        if self._nav_is_complete():
            self._advance_waypoint()

    def _tick_approach(self) -> None:
        if not self._nav_is_complete():
            return
        # Arrived near target
        self._state = State.INTERACT
        self._interact_start = time.monotonic()
        self._speak_to_target(self._current_target)

    def _tick_interact(self) -> None:
        if time.monotonic() - self._interact_start < INTERACT_WAIT_S:
            return
        # Mark visited and check if mission is complete
        t = self._current_target
        if t.kind == 'face':
            self._faces_found += 1
        else:
            self._rings_found += 1

        self.get_logger().info(
            f'Visited {t.kind}. Total: {self._faces_found} faces, {self._rings_found} rings.'
        )

        if self._faces_found >= TOTAL_FACES and self._rings_found >= TOTAL_RINGS:
            self._state = State.DONE
            self._speak('Mission complete. All targets found.')
            self.get_logger().info('DONE — all targets found.')
            return

        self._current_target = None
        self._state = State.EXPLORE
        self._advance_waypoint()

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def _send_approach_goal(self, target: Target) -> None:
        goal = self._make_approach_pose(target)
        self._send_nav_goal(goal)

    def _advance_waypoint(self) -> None:
        wp = EXPLORATION_WAYPOINTS[self._waypoint_idx % len(EXPLORATION_WAYPOINTS)]
        self._waypoint_idx += 1
        goal = self._make_pose(*wp)
        self.get_logger().info(f'Navigating to waypoint ({wp[0]:.1f}, {wp[1]:.1f})')
        self._send_nav_goal(goal)

    def _send_nav_goal(self, pose: PoseStamped) -> None:
        self._nav_result_future = None
        self._nav_goal_handle = None
        self._nav_sent_time = time.monotonic()

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
        self._nav_goal_handle = handle
        self._nav_result_future = handle.get_result_async()

    def _nav_is_complete(self) -> bool:
        if self._nav_result_future is None:
            # No nav in flight — treat as complete so FSM can proceed
            return True
        if not self._nav_result_future.done():
            # Timeout guard
            if (self._nav_sent_time and
                    time.monotonic() - self._nav_sent_time > WAYPOINT_TIMEOUT_S):
                self.get_logger().warn('Waypoint timed out; cancelling.')
                self._cancel_nav()
                return True
            return False
        status = self._nav_result_future.result().status
        self._nav_result_future = None
        return True  # complete (succeeded or failed — move on)

    def _cancel_nav(self) -> None:
        if self._nav_goal_handle is not None:
            self._nav_goal_handle.cancel_goal_async()
            self._nav_goal_handle = None
        self._nav_result_future = None

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def _make_pose(self, x: float, y: float, yaw: float) -> PoseStamped:
        ps = PoseStamped()
        ps.header.frame_id = 'map'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation = self._yaw_to_quat(yaw)
        return ps

    def _make_approach_pose(self, target: Target) -> PoseStamped:
        """Compute a pose APPROACH_DISTANCE metres away from target, facing it."""
        dx = target.x - self._current_pose_x
        dy = target.y - self._current_pose_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            dist = 0.01
        # Unit vector toward target
        ux, uy = dx / dist, dy / dist
        # Approach point
        ax = target.x - ux * APPROACH_DISTANCE
        ay = target.y - uy * APPROACH_DISTANCE
        yaw = math.atan2(uy, ux)
        return self._make_pose(ax, ay, yaw)

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
            colour = target.colour if target.colour else 'unknown'
            self._speak(f'I found a {colour} ring.')

    def _speak(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._speech_pub.publish(msg)

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def _already_visited(self, kind: str, x: float, y: float,
                          threshold: float = 1.0) -> bool:
        """True if a target of this kind near (x, y) is already pending or visited."""
        for t in self._pending_targets:
            if t.kind == kind:
                if math.sqrt((t.x - x) ** 2 + (t.y - y) ** 2) < threshold:
                    return True
        if self._current_target and self._current_target.kind == kind:
            t = self._current_target
            if math.sqrt((t.x - x) ** 2 + (t.y - y) ** 2) < threshold:
                return True
        return False

    # ------------------------------------------------------------------
    # Nav2 readiness
    # ------------------------------------------------------------------
    def _wait_for_nav2(self) -> None:
        from lifecycle_msgs.srv import GetState
        for node_name in ('bt_navigator', 'amcl'):
            svc = f'{node_name}/get_state'
            client = self.create_client(GetState, svc, callback_group=self._cbg)
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {svc}...')
            req = GetState.Request()
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
