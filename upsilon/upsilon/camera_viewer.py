"""Camera viewer — displays compressed RGB plus ring + face detector debug views."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import cv2

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError


QOS_LATEST = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', '/oakd/rgb/preview/image_raw'),
                ('compressed_rgb', False),
            ],
        )
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.compressed_rgb = self.get_parameter('compressed_rgb').get_parameter_value().bool_value
        self.bridge = CvBridge()

        rgb_msg_type = CompressedImage if self.compressed_rgb else Image
        self.create_subscription(
            rgb_msg_type, self.rgb_topic,
            self._rgb_cb, QOS_LATEST)
        self.create_subscription(
            Image, '/ring_detector/debug',
            self._ring_debug_cb, QOS_LATEST)
        self.create_subscription(
            Image, '/ring_detector/threshold',
            self._ring_threshold_cb, QOS_LATEST)
        self.create_subscription(
            Image, '/ring_detector/contour',
            self._ring_contour_cb, QOS_LATEST)
        self.create_subscription(
            Image, '/face_detector/debug',
            self._face_debug_cb, QOS_LATEST)

        for name in ['Camera POV', 'Ring Detection', 'Threshold', 'Contours',
                     'Face Detection']:
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 640, 480)

        self.create_timer(0.03, lambda: cv2.waitKey(1))
        self.get_logger().info('Camera viewer ready — 5 windows.')

    def _rgb_cb(self, msg: Image | CompressedImage) -> None:
        try:
            if isinstance(msg, CompressedImage):
                frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Camera POV', frame)

    def _ring_debug_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Ring Detection', frame)

    def _ring_threshold_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except CvBridgeError:
            return
        cv2.imshow('Threshold', frame)

    def _ring_contour_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Contours', frame)

    def _face_debug_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Face Detection', frame)


def main():
    rclpy.init()
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
