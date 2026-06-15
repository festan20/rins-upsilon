"""Visualizer node — opens OpenCV windows for debugging.

Only subscribes to lightweight/local topics to avoid WiFi bandwidth issues.
The heavy camera topics (depth, RGB) are handled by the detector nodes;
we just display their debug output.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import cv2

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

WINDOW_W = 640
WINDOW_H = 480

QOS_LATEST = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', '/oakd/rgb/preview/image_raw'),
                ('compressed_rgb', False),
                ('top_rgb_topic', '/top_camera/rgb/preview/image_raw'),
                ('top_compressed_rgb', False),
                ('face_debug_topic', '/face_detector/debug'),
                ('ring_debug_topic', '/ring_detector/debug'),
                ('cylinder_debug_topic', ''),
                ('tile_result_topic', '/tile_detection/result'),
                ('tile_debug_topic', '/tile_detection/debug'),
            ],
        )
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.compressed_rgb = self.get_parameter('compressed_rgb').get_parameter_value().bool_value
        self.top_rgb_topic = self.get_parameter('top_rgb_topic').get_parameter_value().string_value
        self.top_compressed_rgb = self.get_parameter('top_compressed_rgb').get_parameter_value().bool_value
        self.face_debug_topic = self.get_parameter('face_debug_topic').get_parameter_value().string_value
        self.ring_debug_topic = self.get_parameter('ring_debug_topic').get_parameter_value().string_value
        self.cylinder_debug_topic = self.get_parameter('cylinder_debug_topic').get_parameter_value().string_value
        self.tile_result_topic = self.get_parameter('tile_result_topic').get_parameter_value().string_value
        self.tile_debug_topic = self.get_parameter('tile_debug_topic').get_parameter_value().string_value
        self.bridge = CvBridge()

        # Only subscribe to compressed RGB (small) and local debug topics
        rgb_msg_type = CompressedImage if self.compressed_rgb else Image
        top_rgb_msg_type = CompressedImage if self.top_compressed_rgb else Image
        self.create_subscription(
            rgb_msg_type, self.rgb_topic,
            self._rgb_cb, QOS_LATEST)
        self.create_subscription(
            top_rgb_msg_type, self.top_rgb_topic,
            self._top_rgb_cb, QOS_LATEST)
        self.create_subscription(
            Image, self.face_debug_topic,
            self._face_debug_cb, QOS_LATEST)
        self.create_subscription(
            Image, self.ring_debug_topic,
            self._ring_debug_cb, QOS_LATEST)

        window_names = ['Camera POV', 'Top Camera POV', 'Face Detection', 'Ring Detection']
        if self.cylinder_debug_topic:
            self.create_subscription(
                Image, self.cylinder_debug_topic,
                self._cylinder_debug_cb, QOS_LATEST)
            window_names.append('Cylinder Detection')
        if self.tile_result_topic:
            self.create_subscription(
                Image, self.tile_result_topic,
                self._tile_result_cb, QOS_LATEST)
            window_names.append('Tile Result')
        if self.tile_debug_topic:
            self.create_subscription(
                Image, self.tile_debug_topic,
                self._tile_debug_cb, QOS_LATEST)
            window_names.append('Tile Debug')

        # Create windows
        for name in window_names:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, WINDOW_W, WINDOW_H)

        self.create_timer(0.03, self._gui_tick)
        self.get_logger().info(f'Visualizer ready — {len(window_names)} windows.')

    def _rgb_cb(self, msg: Image | CompressedImage) -> None:
        try:
            if isinstance(msg, CompressedImage):
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Camera POV', frame)

    def _top_rgb_cb(self, msg: Image | CompressedImage) -> None:
        try:
            if isinstance(msg, CompressedImage):
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Top Camera POV', frame)

    def _face_debug_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Face Detection', frame)

    def _ring_debug_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Ring Detection', frame)

    def _cylinder_debug_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Cylinder Detection', frame)

    def _tile_result_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Tile Result', frame)

    def _tile_debug_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Tile Debug', frame)

    def _gui_tick(self) -> None:
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = VisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
