"""Blue line detector for top-camera based following.

Publishes lightweight signals for a follower state machine:
  - /blue_line/center_error      (std_msgs/Float32)  in [-1, 1]
  - /blue_line/line_visible       (std_msgs/Bool)
  - /blue_line/dead_end           (std_msgs/Bool)
  - /blue_line/branch_offsets     (std_msgs/Float32MultiArray), left->right
  - /blue_line/junction_candidates (geometry_msgs/PoseArray), map frame when TF available
  - /blue_line/debug              (sensor_msgs/Image)

The detector is intentionally simple and robust:
  - HSV thresholding for blue
  - morphological cleanup
  - row-scan segmentation for steering / branch detection
"""

import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped, Pose, PoseArray
from std_msgs.msg import Bool, Float32, Float32MultiArray

from upsilon.perception_utils import decode_depth_message, DepthCameraGeometry, TF2Helper


class BlueLineDetectorNode(Node):
    def __init__(self):
        super().__init__('blue_line_detector')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', '/top_camera/rgb/preview/image_raw'),
                ('depth_topic', '/top_camera/rgb/preview/depth'),
                ('camera_info_topic', '/top_camera/rgb/preview/camera_info'),
                ('compressed_topics', False),
                ('hsv_h_min', 95),
                ('hsv_h_max', 130),
                ('hsv_s_min', 70),
                ('hsv_s_max', 255),
                ('hsv_v_min', 50),
                ('hsv_v_max', 255),
                ('follow_scan_ratio', 0.85),
                ('junction_scan_ratio', 0.55),
                ('min_segment_width_px', 12),
                ('process_hz', 12.0),
                ('dead_end_timeout_sec', 1.5),
            ],
        )

        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.compressed_topics = self.get_parameter('compressed_topics').get_parameter_value().bool_value

        self.hsv_lo = np.array([
            self.get_parameter('hsv_h_min').get_parameter_value().integer_value,
            self.get_parameter('hsv_s_min').get_parameter_value().integer_value,
            self.get_parameter('hsv_v_min').get_parameter_value().integer_value,
        ], dtype=np.uint8)
        self.hsv_hi = np.array([
            self.get_parameter('hsv_h_max').get_parameter_value().integer_value,
            self.get_parameter('hsv_s_max').get_parameter_value().integer_value,
            self.get_parameter('hsv_v_max').get_parameter_value().integer_value,
        ], dtype=np.uint8)

        self.follow_scan_ratio = self.get_parameter('follow_scan_ratio').get_parameter_value().double_value
        self.junction_scan_ratio = self.get_parameter('junction_scan_ratio').get_parameter_value().double_value
        self.min_segment_width_px = self.get_parameter('min_segment_width_px').get_parameter_value().integer_value
        process_hz = self.get_parameter('process_hz').get_parameter_value().double_value
        self.dead_end_timeout_sec = self.get_parameter('dead_end_timeout_sec').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.depth_cam = DepthCameraGeometry(patch_radius=2)
        self.tf2 = TF2Helper(self)

        self._latest_depth_msg = None
        self._depth_frame_id = 'camera_depth_optical_frame'
        self._last_cx = None
        self._last_visible_time = 0.0
        self._last_process_time = 0.0
        self._process_interval = 1.0 / max(1.0, process_hz)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        rgb_msg_type = CompressedImage if self.compressed_topics else Image
        depth_msg_type = CompressedImage if self.compressed_topics else Image

        self.create_subscription(rgb_msg_type, self.rgb_topic, self._rgb_cb, qos)
        self.create_subscription(depth_msg_type, self.depth_topic, self._depth_cb, qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._caminfo_cb, qos)

        self.center_error_pub = self.create_publisher(Float32, '/blue_line/center_error', 10)
        self.visible_pub = self.create_publisher(Bool, '/blue_line/line_visible', 10)
        self.dead_end_pub = self.create_publisher(Bool, '/blue_line/dead_end', 10)
        self.branch_offsets_pub = self.create_publisher(Float32MultiArray, '/blue_line/branch_offsets', 10)
        self.junction_candidates_pub = self.create_publisher(PoseArray, '/blue_line/junction_candidates', 10)
        self.debug_pub = self.create_publisher(Image, '/blue_line/debug', 10)

        self.get_logger().info('Blue line detector ready.')

    def _caminfo_cb(self, msg: CameraInfo) -> None:
        self.depth_cam.update_intrinsics(msg)
        self._depth_frame_id = msg.header.frame_id

    def _depth_cb(self, msg: Image | CompressedImage) -> None:
        self._latest_depth_msg = msg

    def _rgb_cb(self, msg: Image | CompressedImage) -> None:
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._last_process_time < self._process_interval:
            return
        self._last_process_time = now

        try:
            if isinstance(msg, CompressedImage):
                bgr = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            else:
                bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'RGB convert failed: {e}')
            return

        if self._latest_depth_msg is not None:
            depth = decode_depth_message(self._latest_depth_msg, self.bridge)
            if depth is not None:
                self.depth_cam.update_depth(depth)

        self._process_frame(bgr)

    @staticmethod
    def _extract_segments(mask_row: np.ndarray, min_width: int) -> list[tuple[int, int, int]]:
        segments = []
        in_seg = False
        start = 0
        for i, v in enumerate(mask_row):
            if v and not in_seg:
                in_seg = True
                start = i
            elif not v and in_seg:
                end = i - 1
                if end - start + 1 >= min_width:
                    segments.append((start, end, (start + end) // 2))
                in_seg = False
        if in_seg:
            end = len(mask_row) - 1
            if end - start + 1 >= min_width:
                segments.append((start, end, (start + end) // 2))
        return segments

    def _publish_candidates(self, y_scan: int, centers: list[int]) -> None:
        out = PoseArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'map'

        for cx in centers:
            pt_cam = self.depth_cam.get_point(cx, y_scan)
            if pt_cam is None:
                continue

            ps = PointStamped()
            ps.header.stamp = out.header.stamp
            ps.header.frame_id = self._depth_frame_id
            ps.point.x = pt_cam[0]
            ps.point.y = pt_cam[1]
            ps.point.z = pt_cam[2]

            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                continue

            p = Pose()
            p.position.x = ps_map.point.x
            p.position.y = ps_map.point.y
            p.position.z = 0.0
            p.orientation.w = 1.0
            out.poses.append(p)

        self.junction_candidates_pub.publish(out)

    def _process_frame(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lo, self.hsv_hi)

        # Morphology tuned for line continuity.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        y_follow = max(0, min(h - 1, int(h * self.follow_scan_ratio)))
        y_junc = max(0, min(h - 1, int(h * self.junction_scan_ratio)))

        follow_segments = self._extract_segments(mask[y_follow, :] > 0, self.min_segment_width_px)
        line_visible = len(follow_segments) > 0

        center_error = 0.0
        chosen_cx = None
        if line_visible:
            if self._last_cx is None:
                chosen = min(follow_segments, key=lambda s: abs(s[2] - (w // 2)))
            else:
                chosen = min(follow_segments, key=lambda s: abs(s[2] - self._last_cx))
            chosen_cx = chosen[2]
            self._last_cx = chosen_cx
            center_error = float((chosen_cx - (w / 2.0)) / (w / 2.0))
            self._last_visible_time = self.get_clock().now().nanoseconds / 1e9

        now = self.get_clock().now().nanoseconds / 1e9
        dead_end = (now - self._last_visible_time) > self.dead_end_timeout_sec

        junc_segments = self._extract_segments(mask[y_junc, :] > 0, self.min_segment_width_px)
        branch_centers = [s[2] for s in junc_segments]
        branch_offsets = [float((cx - (w / 2.0)) / (w / 2.0)) for cx in branch_centers]

        # Publish primary control signals.
        self.center_error_pub.publish(Float32(data=float(max(-1.0, min(1.0, center_error)))))
        self.visible_pub.publish(Bool(data=bool(line_visible)))
        self.dead_end_pub.publish(Bool(data=bool(dead_end)))
        self.branch_offsets_pub.publish(Float32MultiArray(data=branch_offsets))

        if len(branch_centers) >= 2:
            self._publish_candidates(y_junc, branch_centers)

        # Debug image overlay.
        dbg = bgr.copy()
        cv2.line(dbg, (0, y_follow), (w - 1, y_follow), (0, 255, 255), 1)
        cv2.line(dbg, (0, y_junc), (w - 1, y_junc), (0, 200, 0), 1)
        cv2.line(dbg, (w // 2, 0), (w // 2, h - 1), (255, 255, 255), 1)

        for seg in follow_segments:
            cv2.line(dbg, (seg[0], y_follow), (seg[1], y_follow), (255, 0, 0), 2)
        for seg in junc_segments:
            cv2.line(dbg, (seg[0], y_junc), (seg[1], y_junc), (0, 0, 255), 2)

        if chosen_cx is not None:
            cv2.circle(dbg, (chosen_cx, y_follow), 5, (0, 255, 0), -1)

        txt = f'visible={line_visible} dead_end={dead_end} branches={len(branch_centers)} err={center_error:+.2f}'
        cv2.putText(dbg, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
        except CvBridgeError:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = BlueLineDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
