"""Blue line follower with left-first junction policy.

This node consumes outputs from blue_line_detector and drives /cmd_vel.
Policy summary:
    - At each junction: stop briefly, rotate ~90° left in-place, then scan
      right slowly until the blue line is reacquired.
    - At dead end without nearby face: rotate 180° and continue following.
    - At dead end with nearby face: stop and mark blue-line section complete.
"""

from collections import deque
from dataclasses import dataclass, field
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from std_msgs.msg import Bool, Float32, Float32MultiArray
from geometry_msgs.msg import TwistStamped, PoseArray, PoseWithCovarianceStamped, PointStamped
from std_srvs.srv import SetBool


@dataclass
class JunctionState:
    jid: int
    x: float
    y: float
    total_branches: int
    pending: deque = field(default_factory=deque)
    branch_points: dict[int, tuple[float, float]] = field(default_factory=dict)


class BlueLineFollowerNode(Node):
    def __init__(self):
        super().__init__('blue_line_follower')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('cmd_vel_topic', '/cmd_vel_nav'),
                ('linear_speed', 0.24),
                ('max_angular_speed', 0.8),
                ('k_p', 1.3),
                ('control_hz', 12.0),
                ('lost_line_timeout_sec', 1.0),
                ('blocked_dead_end_timeout_sec', 2.6),
                ('blocked_dead_end_min_progress_m', 0.03),
                ('blocked_dead_end_min_cmd_lin', 0.14),
                ('blocked_dead_end_max_abs_ang', 0.22),
                ('junction_pause_sec', 0.5),
                ('junction_rotate_speed', 0.6),
                ('junction_rotate_sec', 1.4),
                ('junction_scan_speed', 0.3),
                ('junction_scan_timeout_sec', 4.0),
                ('junction_forward_sec', 2.0),
                ('rotate_speed', 0.6),
                ('rotate_duration_sec', 3.2),
                ('junction_match_dist', 0.45),
                ('junction_rearm_dist', 0.35),
                ('face_stop_enabled', True),
                ('face_stop_radius_m', 0.8),
                ('face_max_age_sec', 30.0),
                ('active', False),
            ],
        )

        def _get_double(name):
            return self.get_parameter(name).get_parameter_value().double_value

        def _get_bool(name):
            return self.get_parameter(name).get_parameter_value().bool_value

        def _get_str(name):
            return self.get_parameter(name).get_parameter_value().string_value

        self.linear_speed = _get_double('linear_speed')
        self.max_angular_speed = _get_double('max_angular_speed')
        self.k_p = _get_double('k_p')
        self.cmd_vel_topic = _get_str('cmd_vel_topic')
        self.lost_line_timeout_sec = _get_double('lost_line_timeout_sec')
        self.blocked_dead_end_timeout_sec = _get_double('blocked_dead_end_timeout_sec')
        self.blocked_dead_end_min_progress_m = _get_double('blocked_dead_end_min_progress_m')
        self.blocked_dead_end_min_cmd_lin = _get_double('blocked_dead_end_min_cmd_lin')
        self.blocked_dead_end_max_abs_ang = _get_double('blocked_dead_end_max_abs_ang')
        self.junction_pause_sec = _get_double('junction_pause_sec')
        self.junction_rotate_speed = _get_double('junction_rotate_speed')
        self.junction_rotate_sec = _get_double('junction_rotate_sec')
        self.junction_scan_speed = _get_double('junction_scan_speed')
        self.junction_scan_timeout_sec = _get_double('junction_scan_timeout_sec')
        self.junction_forward_sec = _get_double('junction_forward_sec')
        self.rotate_speed = _get_double('rotate_speed')
        self.rotate_duration_sec = _get_double('rotate_duration_sec')
        self.junction_match_dist = _get_double('junction_match_dist')
        self.junction_rearm_dist = _get_double('junction_rearm_dist')
        self.face_stop_enabled = _get_bool('face_stop_enabled')
        self.face_stop_radius_m = _get_double('face_stop_radius_m')
        self.face_max_age_sec = _get_double('face_max_age_sec')
        self.active = _get_bool('active')

        self.center_error = 0.0
        self.line_visible = False
        self.dead_end = False
        self.branch_offsets: list[float] = []

        self._last_line_time = 0.0
        self._prev_dead_end = False

        self._state = 'FOLLOW' if self.active else 'IDLE'
        self._rotate_until = 0.0
        self._junction_pause_until = 0.0
        self._junction_rotate_until = 0.0
        self._junction_scan_until = 0.0
        self._junction_forward_until = 0.0

        self._pose_x = 0.0
        self._pose_y = 0.0
        self._have_pose = False

        self._junctions: dict[int, JunctionState] = {}
        self._next_jid = 0
        self._last_junction_trigger_pos = None
        self._seen_any_junction = False
        self._blocked_window_start_t = 0.0
        self._blocked_window_start_pose: tuple[float, float] | None = None
        self._face_tracks: dict[int, tuple[float, float, float]] = {}

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Float32, '/blue_line/center_error', self._center_error_cb, qos)
        self.create_subscription(Bool, '/blue_line/line_visible', self._line_visible_cb, qos)
        self.create_subscription(Bool, '/blue_line/dead_end', self._dead_end_cb, qos)
        self.create_subscription(Float32MultiArray, '/blue_line/branch_offsets', self._branch_offsets_cb, qos)
        self.create_subscription(PoseArray, '/blue_line/junction_candidates', self._junction_cb, qos)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self._amcl_cb, qos)
        self.create_subscription(PointStamped, '/detected_faces_task2', self._face_cb, qos)

        self.cmd_pub = self.create_publisher(TwistStamped, self.cmd_vel_topic, 10)
        self._enable_srv = self.create_service(SetBool, '/blue_line/set_active', self._set_active_cb)
        self.add_on_set_parameters_callback(self._on_param_set)

        control_hz = self.get_parameter('control_hz').get_parameter_value().double_value
        self.create_timer(1.0 / max(1.0, control_hz), self._tick)

        self.get_logger().info(
            f'Blue line follower ready. active={self.active} cmd_vel_topic={self.cmd_vel_topic}'
        )

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def _set_active(self, enabled: bool) -> None:
        self.active = bool(enabled)
        self._state = 'FOLLOW' if self.active else 'IDLE'
        self._blocked_window_start_pose = None
        self._blocked_window_start_t = 0.0
        self._junction_pause_until = 0.0
        self._junction_rotate_until = 0.0
        self._junction_scan_until = 0.0
        self._junction_forward_until = 0.0
        if not self.active:
            self._publish_cmd(0.0, 0.0)

    def _set_active_cb(self, request: SetBool.Request, response: SetBool.Response):
        self._set_active(request.data)
        self.set_parameters([Parameter('active', value=self.active)])
        response.success = True
        response.message = f'blue_line_follower active={self.active}'
        self.get_logger().info(response.message)
        return response

    def _on_param_set(self, params):
        for p in params:
            if p.name == 'active' and p.type_ == Parameter.Type.BOOL:
                self._set_active(bool(p.value))
        result = SetParametersResult()
        result.successful = True
        return result

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _center_error_cb(self, msg: Float32) -> None:
        self.center_error = float(max(-1.0, min(1.0, msg.data)))

    def _line_visible_cb(self, msg: Bool) -> None:
        self.line_visible = bool(msg.data)
        if self.line_visible:
            self._last_line_time = self.get_clock().now().nanoseconds / 1e9

    def _dead_end_cb(self, msg: Bool) -> None:
        self.dead_end = bool(msg.data)

    def _branch_offsets_cb(self, msg: Float32MultiArray) -> None:
        self.branch_offsets = [float(v) for v in msg.data]

    def _amcl_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self._pose_x = msg.pose.pose.position.x
        self._pose_y = msg.pose.pose.position.y
        self._have_pose = True

    def _face_cb(self, msg: PointStamped) -> None:
        parts = msg.header.frame_id.split('/')
        if len(parts) < 3 or parts[0] != 'map':
            return
        try:
            track_id = int(parts[1])
        except ValueError:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        self._face_tracks[track_id] = (msg.point.x, msg.point.y, now)

    # ------------------------------------------------------------------
    # Junction logic
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
        return math.hypot(a_x - b_x, a_y - b_y)

    def _nearest_junction_id(self, x: float, y: float, max_dist: float) -> int | None:
        best_id = None
        best_d = 1e9
        for jid, st in self._junctions.items():
            d = self._dist(x, y, st.x, st.y)
            if d < best_d and d <= max_dist:
                best_d = d
                best_id = jid
        return best_id

    def _create_or_update_junction(
        self,
        x: float,
        y: float,
        n_branches: int,
        branch_points: list[tuple[float, float]] | None = None,
    ) -> int:
        jid = self._nearest_junction_id(x, y, self.junction_match_dist)
        if jid is not None:
            st = self._junctions[jid]
            if branch_points:
                for i, pt in enumerate(branch_points):
                    st.branch_points[i] = pt
            if n_branches > st.total_branches:
                st.total_branches = n_branches
            return jid

        jid = self._next_jid
        self._next_jid += 1
        st = JunctionState(jid=jid, x=x, y=y, total_branches=n_branches)
        if branch_points:
            for i, pt in enumerate(branch_points):
                st.branch_points[i] = pt
        self._junctions[jid] = st
        self.get_logger().info(f'New junction #{jid} with {n_branches} branches.')
        return jid

    def _trigger_junction(self, jid: int) -> None:
        """Begin junction handling: stop, rotate left, scan for line."""
        now = self.get_clock().now().nanoseconds / 1e9
        self._junction_pause_until = now + self.junction_pause_sec
        self._state = 'JUNCTION_PAUSE'
        self.get_logger().info(
            f'Junction #{jid}: stopping then rotating left ~90°.'
        )

    def _junction_cb(self, msg: PoseArray) -> None:
        if not self.active:
            return
        if self._state not in ('FOLLOW',):
            return
        if not self._have_pose:
            return
        if len(msg.poses) < 2:
            return

        # De-bounce: ignore if still near the last triggered junction.
        if self._last_junction_trigger_pos is not None:
            d_last = self._dist(
                self._pose_x, self._pose_y,
                self._last_junction_trigger_pos[0], self._last_junction_trigger_pos[1],
            )
            if d_last < self.junction_rearm_dist:
                return

        branch_points = [(p.position.x, p.position.y) for p in msg.poses]
        jid = self._create_or_update_junction(
            self._pose_x, self._pose_y, len(msg.poses), branch_points=branch_points,
        )
        self._seen_any_junction = True
        self._last_junction_trigger_pos = (self._pose_x, self._pose_y)
        self._trigger_junction(jid)

    # ------------------------------------------------------------------
    # Dead-end logic
    # ------------------------------------------------------------------

    def _check_blocked_dead_end(self, now: float, cmd_lin: float, cmd_ang: float) -> bool:
        """Treat sustained no-progress while commanding forward as a dead-end."""
        tracking_enabled = (
            self.active
            and self._have_pose
            and self.line_visible
            and self._state == 'FOLLOW'
            and cmd_lin >= self.blocked_dead_end_min_cmd_lin
            and abs(cmd_ang) <= self.blocked_dead_end_max_abs_ang
        )
        if not tracking_enabled:
            self._blocked_window_start_pose = None
            self._blocked_window_start_t = 0.0
            return False

        if self._blocked_window_start_pose is None:
            self._blocked_window_start_pose = (self._pose_x, self._pose_y)
            self._blocked_window_start_t = now
            return False

        progress = self._dist(
            self._pose_x, self._pose_y,
            self._blocked_window_start_pose[0], self._blocked_window_start_pose[1],
        )
        if progress >= self.blocked_dead_end_min_progress_m:
            self._blocked_window_start_pose = (self._pose_x, self._pose_y)
            self._blocked_window_start_t = now
            return False

        return (now - self._blocked_window_start_t) >= self.blocked_dead_end_timeout_sec

    def _start_dead_end_recovery(self) -> None:
        if self._state == 'COMPLETE':
            return
        self._blocked_window_start_pose = None
        self._blocked_window_start_t = 0.0
        if self.face_stop_enabled and self._has_recent_face_nearby():
            self._state = 'COMPLETE'
            self._publish_cmd(0.0, 0.0)
            self.get_logger().info('Reached dead end with nearby face: blue-line section complete.')
            return

        now = self.get_clock().now().nanoseconds / 1e9
        self._state = 'ROTATING'
        self._rotate_until = now + self.rotate_duration_sec
        self.get_logger().info('Dead end without nearby face: rotating to reverse direction.')

    def _has_recent_face_nearby(self) -> bool:
        if not self._have_pose:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        for fx, fy, seen_t in self._face_tracks.values():
            if (now - seen_t) > self.face_max_age_sec:
                continue
            if self._dist(self._pose_x, self._pose_y, fx, fy) <= self.face_stop_radius_m:
                return True
        return False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish_cmd(self, lin: float, ang: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(lin)
        msg.twist.angular.z = float(ang)
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9

        # Fallback junction detection from image-space branch offsets (top camera).
        if self.active and self._state == 'FOLLOW' and self._have_pose:
            if len(self.branch_offsets) >= 2:
                allow = True
                if self._last_junction_trigger_pos is not None:
                    d = self._dist(
                        self._pose_x, self._pose_y,
                        self._last_junction_trigger_pos[0], self._last_junction_trigger_pos[1],
                    )
                    allow = d >= self.junction_rearm_dist
                if allow:
                    jid = self._create_or_update_junction(
                        self._pose_x, self._pose_y, len(self.branch_offsets),
                    )
                    self._seen_any_junction = True
                    self._last_junction_trigger_pos = (self._pose_x, self._pose_y)
                    self._trigger_junction(jid)
                    return

        if self._state in ('IDLE', 'COMPLETE'):
            self._publish_cmd(0.0, 0.0)
            self._blocked_window_start_pose = None
            self._blocked_window_start_t = 0.0
            return

        # 180° rotation for dead-end reversal.
        if self._state == 'ROTATING':
            if now < self._rotate_until:
                self._publish_cmd(0.0, self.rotate_speed)
                return
            self._state = 'FOLLOW'

        # Junction handling: stop → rotate left → scan right for line.
        if self._state == 'JUNCTION_PAUSE':
            if now < self._junction_pause_until:
                self._publish_cmd(0.0, 0.0)
                return
            self._junction_rotate_until = now + self.junction_rotate_sec
            self._state = 'JUNCTION_ROTATE'

        if self._state == 'JUNCTION_ROTATE':
            if now < self._junction_rotate_until:
                self._publish_cmd(0.0, self.junction_rotate_speed)  # CCW = left
                return
            self._junction_scan_until = now + self.junction_scan_timeout_sec
            self._state = 'JUNCTION_SCAN'
            self.get_logger().info('Junction rotate done — scanning right for blue line.')

        if self._state == 'JUNCTION_SCAN':
            if self.line_visible:
                self.get_logger().info('Junction scan: blue line reacquired — driving forward.')
                self._junction_forward_until = now + self.junction_forward_sec
                self._state = 'JUNCTION_FORWARD'
            elif now < self._junction_scan_until:
                self._publish_cmd(0.0, -self.junction_scan_speed)  # CW = right scan
                return
            else:
                self.get_logger().warn(
                    'Junction scan timed out without finding blue line — driving forward anyway.'
                )
                self._junction_forward_until = now + self.junction_forward_sec
                self._state = 'JUNCTION_FORWARD'

        # Follow the line forward briefly after a junction before re-arming detection.
        if self._state == 'JUNCTION_FORWARD':
            if now < self._junction_forward_until:
                err = self.center_error
                ang = -self.k_p * err
                ang = max(-self.max_angular_speed, min(self.max_angular_speed, ang))
                self._publish_cmd(self.linear_speed * max(0.2, 1.0 - abs(err)), ang)
                return
            self._state = 'FOLLOW'

        # FOLLOW control.
        err = self.center_error
        ang = -self.k_p * err
        ang = max(-self.max_angular_speed, min(self.max_angular_speed, ang))

        lin_scale = max(0.2, 1.0 - abs(err))
        lin = self.linear_speed * lin_scale

        blocked_dead_end = self._check_blocked_dead_end(now, lin, ang)
        dead_end_event = self.dead_end or blocked_dead_end
        if dead_end_event:
            if self.face_stop_enabled and self._has_recent_face_nearby():
                self._state = 'COMPLETE'
                self._publish_cmd(0.0, 0.0)
                self.get_logger().info('Reached dead end with nearby face: blue-line section complete.')
                self._prev_dead_end = True
                return

            if not self._prev_dead_end:
                if blocked_dead_end and not self.dead_end:
                    self.get_logger().info('Blocked forward progress detected: treating as dead end.')
                self._start_dead_end_recovery()
                self._prev_dead_end = True
                return
        self._prev_dead_end = dead_end_event

        if not self.line_visible and (now - self._last_line_time) > self.lost_line_timeout_sec:
            self._publish_cmd(0.0, -0.25)
            return

        self._publish_cmd(lin, ang)


def main(args=None):
    rclpy.init(args=args)
    node = BlueLineFollowerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
