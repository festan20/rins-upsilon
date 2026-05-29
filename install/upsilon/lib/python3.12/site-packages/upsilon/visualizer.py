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
        self.bridge = CvBridge()

        # Only subscribe to compressed RGB (small) and local debug topics
        self.create_subscription(
            CompressedImage, '/gemini/color/image_raw/compressed',
            self._rgb_cb, QOS_LATEST)
        self.create_subscription(
            Image, '/face_detector/debug',
            self._face_debug_cb, QOS_LATEST)
        self.create_subscription(
            Image, '/ring_detector/debug',
            self._ring_debug_cb, QOS_LATEST)

        # Create windows
        for name in ['Camera POV', 'Face Detection', 'Ring Detection']:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, WINDOW_W, WINDOW_H)

        self.create_timer(0.03, self._gui_tick)
        self.get_logger().info('Visualizer ready — 3 windows.')

    def _rgb_cb(self, msg: CompressedImage) -> None:
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Camera POV', frame)

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
