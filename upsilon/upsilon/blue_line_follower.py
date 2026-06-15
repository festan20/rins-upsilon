"""Blue line follower with left-first junction policy.

This node consumes outputs from blue_line_detector and drives /cmd_vel.
Policy summary:
    - At each junction, always commit to the leftmost branch (robot-local image left).
    - At dead end without nearby face: rotate and continue following.
    - At dead end with nearby face: stop and mark blue-line section complete.

Notes
-----
- Junction identity is map-frame proximity based.
- Steering is visual-servo on center_error with a temporary branch bias.
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


@dataclass
class BranchTask:
    jid: int
    branch_idx: int
    target_x: float | None = None
    target_y: float | None = None


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
                ('branch_commit_sec', 1.2),
                ('junction_pause_sec', 1.0),
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

        self.linear_speed = self.get_parameter('linear_speed').get_parameter_value().double_value
        self.max_angular_speed = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        self.k_p = self.get_parameter('k_p').get_parameter_value().double_value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').get_parameter_value().string_value
        self.lost_line_timeout_sec = self.get_parameter('lost_line_timeout_sec').get_parameter_value().double_value
        self.blocked_dead_end_timeout_sec = self.get_parameter(
            'blocked_dead_end_timeout_sec'
        ).get_parameter_value().double_value
        self.blocked_dead_end_min_progress_m = self.get_parameter(
            'blocked_dead_end_min_progress_m'
        ).get_parameter_value().double_value
        self.blocked_dead_end_min_cmd_lin = self.get_parameter(
            'blocked_dead_end_min_cmd_lin'
        ).get_parameter_value().double_value
        self.blocked_dead_end_max_abs_ang = self.get_parameter(
            'blocked_dead_end_max_abs_ang'
        ).get_parameter_value().double_value
        self.branch_commit_sec = self.get_parameter('branch_commit_sec').get_parameter_value().double_value
        self.junction_pause_sec = self.get_parameter('junction_pause_sec').get_parameter_value().double_value
        self.rotate_speed = self.get_parameter('rotate_speed').get_parameter_value().double_value
        self.rotate_duration_sec = self.get_parameter('rotate_duration_sec').get_parameter_value().double_value
        self.junction_match_dist = self.get_parameter('junction_match_dist').get_parameter_value().double_value
        self.junction_rearm_dist = self.get_parameter('junction_rearm_dist').get_parameter_value().double_value
        self.face_stop_enabled = self.get_parameter('face_stop_enabled').get_parameter_value().bool_value
        self.face_stop_radius_m = self.get_parameter('face_stop_radius_m').get_parameter_value().double_value
        self.face_max_age_sec = self.get_parameter('face_max_age_sec').get_parameter_value().double_value
        self.active = self.get_parameter('active').get_parameter_value().bool_value

        self.center_error = 0.0
        self.line_visible = False
        self.dead_end = False
        self.branch_offsets: list[float] = []

        self._last_line_time = 0.0
        self._prev_dead_end = False

        self._state = 'FOLLOW' if self.active else 'IDLE'
        self._rotate_until = 0.0
        self._junction_pause_until = 0.0
        self._junction_turn_until = 0.0
        self._branch_bias = 0.0
        self._branch_bias_until = 0.0

        self._pose_x = 0.0
        self._pose_y = 0.0
        self._have_pose = False

        self._junctions: dict[int, JunctionState] = {}
        self._next_jid = 0
        self._queue: deque[BranchTask] = deque()
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

    def _set_active(self, enabled: bool) -> None:
        self.active = bool(enabled)
        self._state = 'FOLLOW' if self.active else 'IDLE'
        self._blocked_window_start_pose = None
        self._blocked_window_start_t = 0.0
        self._junction_pause_until = 0.0
        self._junction_turn_until = 0.0
        if not self.active:
            self._publish_cmd(0.0, 0.0)

    def _set_active_cb(self, request: SetBool.Request, response: SetBool.Response):
        self._set_active(request.data)
        # Keep parameter value in sync with service-driven state.
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

    @staticmethod
    def _nearest_branch_index(
        candidates: list[tuple[float, float]],
        target_x: float,
        target_y: float,
    ) -> int | None:
        if not candidates:
            return None
        best_idx = None
        best_d = 1e9
        for i, (cx, cy) in enumerate(candidates):
            d = math.hypot(cx - target_x, cy - target_y)
            if d < best_d:
                best_d = d
                best_idx = i
        return best_idx

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
        for i in range(n_branches):
            if branch_points and i < len(branch_points):
                st.branch_points[i] = branch_points[i]
        self._junctions[jid] = st

        self.get_logger().info(f'New junction #{jid} with {n_branches} branches.')
        return jid

    def _select_branch_at_junction(
        self,
        jid: int,
        prefer_queue: bool,
        branch_points: list[tuple[float, float]] | None = None,
    ) -> int | None:
        del jid, prefer_queue, branch_points
        if len(self.branch_offsets) > 0:
            return 0
        return None

    def _apply_branch_bias(self, branch_idx: int) -> None:
        if not self.branch_offsets:
            # Fallback bias when detector branch offsets are not available yet.
            self._branch_bias = -0.3 + 0.3 * branch_idx
        elif 0 <= branch_idx < len(self.branch_offsets):
            self._branch_bias = self.branch_offsets[branch_idx]
        else:
            self._branch_bias = 0.0
        now = self.get_clock().now().nanoseconds / 1e9
        self._branch_bias_until = now + self.branch_commit_sec

    def _start_junction_pause(self, branch_idx: int) -> None:
        del branch_idx
        now = self.get_clock().now().nanoseconds / 1e9
        self._branch_bias = 0.0
        self._branch_bias_until = 0.0
        self._junction_pause_until = now + self.junction_pause_sec
        self._junction_turn_until = self._junction_pause_until + self.branch_commit_sec
        self._state = 'JUNCTION_PAUSE'

    def _start_junction_turn(self, branch_idx: int) -> None:
        self._apply_branch_bias(branch_idx)
        self._start_junction_pause(branch_idx)

    def _junction_cb(self, msg: PoseArray) -> None:
        if not self.active or self._state == 'IDLE' or self._state == 'COMPLETE':
            return
        if not self._have_pose:
            return
        if len(msg.poses) < 2:
            return

        branch_points = [(p.position.x, p.position.y) for p in msg.poses]

        # De-bounce: avoid retriggering the same junction while standing near it.
        if self._last_junction_trigger_pos is not None:
            d_last = self._dist(
                self._pose_x, self._pose_y,
                self._last_junction_trigger_pos[0], self._last_junction_trigger_pos[1],
            )
            if d_last < self.junction_rearm_dist:
                return

        jid = self._create_or_update_junction(
            self._pose_x,
            self._pose_y,
            len(msg.poses),
            branch_points=branch_points,
        )
        self._seen_any_junction = True
        chosen = 0 if len(msg.poses) >= 2 else None

        if chosen is not None:
            self._start_junction_turn(chosen)
            self.get_logger().info(f'Reached junction #{jid}: taking left branch {chosen}.')

        self._last_junction_trigger_pos = (self._pose_x, self._pose_y)

    def _publish_cmd(self, lin: float, ang: float) -> None:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = float(lin)
        msg.twist.angular.z = float(ang)
        self.cmd_pub.publish(msg)

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
            self._pose_x,
            self._pose_y,
            self._blocked_window_start_pose[0],
            self._blocked_window_start_pose[1],
        )
        if progress >= self.blocked_dead_end_min_progress_m:
            # Robot is progressing, slide window forward.
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
        self.get_logger().info('Dead end without nearby face: rotate and continue.')

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

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds / 1e9

        # Fallback branch detection path when PoseArray junction candidates are sparse.
        # This uses image-space branch offsets and current pose to build/update queue.
        if self.active and self._state in ('FOLLOW', 'BACKTRACK') and self._have_pose:
            if len(self.branch_offsets) >= 2:
                allow_trigger = True
                if self._last_junction_trigger_pos is not None:
                    d_last = self._dist(
                        self._pose_x,
                        self._pose_y,
                        self._last_junction_trigger_pos[0],
                        self._last_junction_trigger_pos[1],
                    )
                    allow_trigger = d_last >= self.junction_rearm_dist

                if allow_trigger:
                    jid = self._create_or_update_junction(
                        self._pose_x,
                        self._pose_y,
                        len(self.branch_offsets),
                    )
                    self._seen_any_junction = True
                    chosen = 0
                    if chosen is not None:
                        self._start_junction_turn(chosen)
                        self.get_logger().info(
                            f'Reached junction #{jid} (fallback): taking left branch {chosen}.'
                        )

                    self._last_junction_trigger_pos = (self._pose_x, self._pose_y)

        # Rising-edge dead-end handling.
        if self._state in ('IDLE', 'COMPLETE'):
            self._publish_cmd(0.0, 0.0)
            self._blocked_window_start_pose = None
            self._blocked_window_start_t = 0.0
            return

        if self._state == 'ROTATING':
            if now < self._rotate_until:
                self._publish_cmd(0.0, self.rotate_speed)
                return
            self._state = 'FOLLOW'

        if self._state == 'JUNCTION_PAUSE':
            if now < self._junction_pause_until:
                self._publish_cmd(0.0, 0.0)
                return
            self._state = 'JUNCTION_TURN'

        if self._state == 'JUNCTION_TURN':
            if now < self._junction_turn_until:
                desired = self._branch_bias if self._branch_bias_until > now else 0.0
                err = self.center_error - desired
                ang = -1.8 * self.k_p * err
                ang = max(-self.max_angular_speed, min(self.max_angular_speed, ang))
                lin = min(self.linear_speed * 0.5, 0.08)
                self._publish_cmd(lin, ang)
                return
            self._state = 'FOLLOW'

        desired = 0.0
        if now < self._branch_bias_until:
            desired = self._branch_bias

        err = self.center_error - desired
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

        # FOLLOW or BACKTRACK control.
        if not self.line_visible and (now - self._last_line_time) > self.lost_line_timeout_sec:
            # Slow-search rotation while trying to reacquire line.
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
