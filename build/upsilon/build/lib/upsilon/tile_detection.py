"""Tile detection node — detects a rectangular tile in the top camera view and
returns a perspective-corrected square image via a ROS2 service.

Service:  /detect_tile  (std_srvs/srv/Trigger)
  On call: finds the tile rectangle in the latest top camera frame,
           warps it to a square, and publishes the result.

Publishes:
  /tile_detection/result  (sensor_msgs/msg/Image) — square perspective-corrected tile
  /tile_detection/debug   (sensor_msgs/msg/Image) — annotated frame with detected corners
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge, CvBridgeError


QOS_LATEST = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

OUTPUT_SIZE = 512  # pixels — side length of the output square


class TileDetectionNode(Node):
    def __init__(self):
        super().__init__('tile_detection')

        self.declare_parameter('rgb_topic', '/top_camera/rgb/preview/image_raw')
        self.declare_parameter('crop_bottom_fraction', 0.25)
        self.declare_parameter('min_area_fraction', 0.02)

        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.crop_bottom = self.get_parameter('crop_bottom_fraction').get_parameter_value().double_value
        self.min_area_frac = self.get_parameter('min_area_fraction').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.latest_frame = None

        self.create_subscription(Image, self.rgb_topic, self._rgb_cb, QOS_LATEST)
        self.result_pub = self.create_publisher(Image, '/tile_detection/result', 10)
        self.debug_pub = self.create_publisher(Image, '/tile_detection/debug', 10)
        self.create_service(Trigger, 'detect_tile', self._detect_tile_cb)

        self.get_logger().info(f'Tile detection ready on {self.rgb_topic}. Call /detect_tile.')

    def _rgb_cb(self, msg: Image) -> None:
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            pass

    def _detect_tile_cb(self, request, response):
        if self.latest_frame is None:
            response.success = False
            response.message = 'No camera frame received yet.'
            return response

        frame = self.latest_frame.copy()
        corners = self._detect_rectangle(frame)

        debug = frame.copy()
        if corners is None:
            cv2.putText(debug, 'No tile detected', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
            response.success = False
            response.message = 'No tile rectangle found in frame.'
            return response

        cv2.polylines(debug, [corners], True, (0, 255, 0), 3)
        for pt in corners:
            cv2.circle(debug, tuple(pt[0]), 8, (0, 0, 255), -1)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))

        warped = self._warp_to_square(frame, corners)
        self.result_pub.publish(self.bridge.cv2_to_imgmsg(warped, 'bgr8'))

        response.success = True
        response.message = 'Tile detected. Result on /tile_detection/result.'
        return response

    def _detect_rectangle(self, frame) -> np.ndarray | None:
        h, w = frame.shape[:2]
        crop_h = int(h * (1.0 - self.crop_bottom))
        roi = frame[:crop_h, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu finds the threshold automatically between dark wall and bright tile
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = w * crop_h * self.min_area_frac
        best, best_area = None, 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4 and area > best_area:
                best, best_area = approx, area

        return best

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Return corners ordered: top-left, top-right, bottom-right, bottom-left."""
        pts = pts.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]     # top-left
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[2] = pts[np.argmax(s)]     # bottom-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect

    def _warp_to_square(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        src = self._order_corners(corners)
        n = OUTPUT_SIZE
        dst = np.array([[0, 0], [n - 1, 0], [n - 1, n - 1], [0, n - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(frame, M, (n, n))


def main():
    rclpy.init()
    node = TileDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
