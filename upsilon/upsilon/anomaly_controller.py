"""Anomaly mission controller.

Sequential flow:
    1. Rotate the top camera to the left  (publish to /arm_command)
    2. Drive every checkpoint in the selected set (red / green). At each one:
         a. navigate to the pose
         b. trigger tile detection      (/detect_tile,   std_srvs/Trigger)
         c. trigger anomaly detection   (/detect_anomaly, std_srvs/Trigger)
            and remember whether an anomaly was found
    3. Print a summary of which checkpoints had anomalies.

Two checkpoint sets are provided below (RED_CHECKPOINTS / GREEN_CHECKPOINTS);
pick one with the `checkpoint_set` parameter ('red' or 'green').

Runs on a MultiThreadedExecutor so action/service futures resolve while the
mission thread blocks (same pattern as controller.py).


ros2 run upsilon anomaly_controller --ros-args -p checkpoint_set:=green
"""

import math
import threading
import time

from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import String
from std_srvs.srv import Trigger

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler


# --------------------------------------------------------------------------
# Tuning parameters
# --------------------------------------------------------------------------
NAV_TIMEOUT_S      = 60.0   # give up on a nav goal after this long
SERVICE_TIMEOUT_S  = 15.0   # give up on a service call after this long
POLL_INTERVAL_S    = 0.3    # sleep between blocking polls
TILE_SETTLE_S      = 0.5    # let the warped tile propagate to the anomaly node
ARM_SETTLE_S       = 4.0    # wait for the arm to finish moving (3s trajectory)

# Arm pose that turns the top camera to the LEFT.
# Joints: [arm_base_joint, arm_shoulder_joint, arm_elbow_joint, arm_wrist_joint]
# The base joint rotates the camera: +1.57 ≈ left, -1.57 ≈ right
# (cf. look_at_belt_left / look_at_belt_right in arm_mover_actions.py).
# The wrist joint tilts the camera pitch: lower value → looks more UPWARD
# (toward the 'up' pose [0,0,0,0]); raise it back toward 2.0 to look down.
ARM_LEFT_COMMAND = 'manual:[1.57, 0.6, 0.5, 1.7]'

# Checkpoints to visit, [x, y, yaw_rad] in the map frame.
# TODO: fill in the real positions for each team's side.
RED_CHECKPOINTS: list[tuple[float, float, float]] = [
    # (x, y, yaw) — yaw = -pi/2 (90° to the right) while inspecting each tile.
    (0.249043807387352,  -4.6911163330078125, -math.pi),
    (-0.2910171151161194, -4.659816265106201, -math.pi),
    (-0.8702130913734436, -4.667640686035156, -math.pi),
    (-1.4024468660354614, -4.722416877746582, -math.pi),
]

GREEN_CHECKPOINTS: list[tuple[float, float, float]] = [
    # (x, y, yaw) — yaw = pi/2 (zahod / west) while inspecting each tile.
    (-4.643223762512207, -2.5365700721740723, math.pi / 2),
    (-4.64479398727417,  -1.973413348197937,  math.pi / 2),
    (-4.686208248138428, -1.2324994802474976, math.pi / 2),
    (-4.792420864105225, -0.533107340335846,  math.pi / 2),
    (-4.832175254821777,  0.19783915579319,   math.pi / 2),
]


class AnomalyControllerNode(Node):
    def __init__(self):
        super().__init__('anomaly_controller')

        self.declare_parameter('checkpoint_set', 'green')
        self.declare_parameter('arm_command', ARM_LEFT_COMMAND)

        self._set_name = self.get_parameter('checkpoint_set').get_parameter_value().string_value
        self._arm_command = self.get_parameter('arm_command').get_parameter_value().string_value

        self._cbg = ReentrantCallbackGroup()

        # Nav2 action client
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                        callback_group=self._cbg)

        # Detector trigger services
        self._tile_client = self.create_client(Trigger, 'detect_tile',
                                                callback_group=self._cbg)
        self._anomaly_client = self.create_client(Trigger, 'detect_anomaly',
                                                   callback_group=self._cbg)

        # Arm command publisher
        self._arm_pub = self.create_publisher(String, '/arm_command', 10)

        # checkpoint index -> dict(x, y, anomaly, message)
        self._results: dict[int, dict] = {}

        # Start mission on a separate thread so callbacks keep firing
        self._mission_thread = threading.Thread(target=self._run_mission, daemon=True)
        self._mission_thread.start()

    # ------------------------------------------------------------------
    # Mission (blocking, runs on its own thread)
    # ------------------------------------------------------------------
    def _checkpoints(self) -> list[tuple[float, float, float]]:
        if self._set_name == 'red':
            return RED_CHECKPOINTS
        if self._set_name == 'green':
            return GREEN_CHECKPOINTS
        self.get_logger().warn(
            f"Unknown checkpoint_set '{self._set_name}'; defaulting to green.")
        return GREEN_CHECKPOINTS

    def _run_mission(self) -> None:
        self._wait_for_nav2()
        self.get_logger().info('Nav2 is up.')

        # Step 1 — turn the top camera to the left
        self.get_logger().info(f'Turning top camera left: {self._arm_command}')
        self._move_arm(self._arm_command)

        checkpoints = self._checkpoints()
        if not checkpoints:
            self.get_logger().warn(
                f"No '{self._set_name}' checkpoints defined — fill in "
                'RED_CHECKPOINTS / GREEN_CHECKPOINTS in anomaly_controller.py.')
            return

        # Step 2 — visit every checkpoint
        self.get_logger().info(
            f'Visiting {len(checkpoints)} {self._set_name} checkpoints.')
        for i, (x, y, yaw) in enumerate(checkpoints):
            self.get_logger().info(
                f'Checkpoint {i + 1}/{len(checkpoints)} → ({x:.2f}, {y:.2f})')

            if not self._navigate_to(x, y, yaw):
                self.get_logger().warn(
                    f'Checkpoint {i + 1} unreachable; recording as skipped.')
                self._results[i] = {'x': x, 'y': y, 'anomaly': None,
                                    'message': 'navigation failed'}
                continue

            anomaly, message = self._inspect_tile()
            self._results[i] = {'x': x, 'y': y, 'anomaly': anomaly,
                                'message': message}

        # Step 3 — summary
        self._print_summary()

    def _inspect_tile(self) -> tuple[bool | None, str]:
        """Trigger tile detection then anomaly detection at the current pose."""
        tile_ok, tile_msg = self._call_trigger(self._tile_client, 'detect_tile')
        if not tile_ok:
            self.get_logger().warn(f'Tile detection failed: {tile_msg}')
            return None, f'tile detection failed: {tile_msg}'

        # Let the warped tile reach the anomaly node before asking it to run.
        time.sleep(TILE_SETTLE_S)

        anomaly, anomaly_msg = self._call_trigger(self._anomaly_client, 'detect_anomaly')
        label = 'ANOMALY' if anomaly else 'okay'
        self.get_logger().info(f'Result: {label} — {anomaly_msg}')
        return anomaly, anomaly_msg

    def _print_summary(self) -> None:
        self.get_logger().info('=== Anomaly inspection summary ===')
        anomalies = 0
        for i in sorted(self._results):
            r = self._results[i]
            if r['anomaly'] is True:
                state = 'ANOMALY'
                anomalies += 1
            elif r['anomaly'] is False:
                state = 'okay'
            else:
                state = 'skipped'
            self.get_logger().info(
                f"  [{i + 1}] ({r['x']:.2f}, {r['y']:.2f}): {state} — {r['message']}")
        self.get_logger().info(
            f'{anomalies} anomaly/anomalies across {len(self._results)} checkpoints.')

    # ------------------------------------------------------------------
    # Arm control
    # ------------------------------------------------------------------
    def _move_arm(self, command: str) -> None:
        msg = String()
        msg.data = command
        self._arm_pub.publish(msg)
        time.sleep(ARM_SETTLE_S)

    # ------------------------------------------------------------------
    # Service helper
    # ------------------------------------------------------------------
    def _call_trigger(self, client, name: str) -> tuple[bool, str]:
        """Call a std_srvs/Trigger service, blocking until it returns."""
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f'Service {name} not available.')
            return False, 'service unavailable'

        future = client.call_async(Trigger.Request())
        t0 = time.monotonic()
        while not future.done():
            time.sleep(POLL_INTERVAL_S)
            if time.monotonic() - t0 > SERVICE_TIMEOUT_S:
                self.get_logger().warn(f'Service {name} timed out.')
                return False, 'timed out'

        result = future.result()
        return result.success, result.message

    # ------------------------------------------------------------------
    # Blocking navigation
    # ------------------------------------------------------------------
    def _navigate_to(self, x: float, y: float, yaw: float) -> bool:
        """Send a nav goal and block until it completes or times out."""
        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Nav2 not available.')
            return False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._make_pose(x, y, yaw)
        send_future = self._nav_client.send_goal_async(goal_msg)

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

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(POLL_INTERVAL_S)
            if time.monotonic() - t0 > NAV_TIMEOUT_S:
                self.get_logger().warn('Nav goal timed out; cancelling.')
                goal_handle.cancel_goal_async()
                return False

        return True

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

    @staticmethod
    def _yaw_to_quat(yaw: float) -> Quaternion:
        q = quaternion_from_euler(0.0, 0.0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    # ------------------------------------------------------------------
    # Nav2 readiness
    # ------------------------------------------------------------------
    def _wait_for_nav2(self) -> None:
        from lifecycle_msgs.srv import GetState
        self.get_logger().info('Waiting for Nav2...')
        for node_name in ('bt_navigator', 'amcl'):
            svc = f'{node_name}/get_state'
            client = self.create_client(GetState, svc, callback_group=self._cbg)
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f'Waiting for {svc}...')
            req = GetState.Request()
            state = ''
            while state != 'active':
                future = client.call_async(req)
                while not future.done():
                    time.sleep(0.2)
                if future.result():
                    state = future.result().current_state.label
                time.sleep(1.0)
            self.get_logger().info(f'{node_name} is active.')


def main():
    rclpy.init()
    node = AnomalyControllerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
