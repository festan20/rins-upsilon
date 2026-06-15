"""QR reader node (Task 2).

Reads QR codes from camera images and publishes decoded instruction text.

Published topics
----------------
/qr_instructions_task2   (std_msgs/String) - newly discovered instruction texts
/qr_reader_task2/debug   (sensor_msgs/Image) - annotated BGR frame
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError


class QrReaderTask2Node(Node):
    def __init__(self):
        super().__init__('qr_reader_task2')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', '/oakd/rgb/preview/image_raw'),
                ('publish_duplicates', False),
            ],
        )
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.publish_duplicates = self.get_parameter(
            'publish_duplicates'
        ).get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.qr_detector = cv2.QRCodeDetector()
        self._seen_texts: set[str] = set()

        self.create_subscription(Image, self.rgb_topic, self._rgb_cb, qos_profile_sensor_data)
        self._instruction_pub = self.create_publisher(String, '/qr_instructions_task2', 10)
        self._debug_pub = self.create_publisher(Image, '/qr_reader_task2/debug', 10)

        self.get_logger().info(f'QR reader (Task 2) ready on {self.rgb_topic}.')

    def _rgb_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')
            return

        debug = frame.copy()

        found_any = False
        try:
            ok, decoded_texts, points, _ = self.qr_detector.detectAndDecodeMulti(frame)
        except Exception as e:
            self.get_logger().error(f'QR decode failed: {e}', throttle_duration_sec=2.0)
            ok, decoded_texts, points = False, (), None

        if ok and decoded_texts:
            for i, text in enumerate(decoded_texts):
                text = text.strip()
                if not text:
                    continue

                found_any = True

                # Draw QR polygon when available.
                if points is not None and i < len(points):
                    poly = points[i].astype(int)
                    cv2.polylines(debug, [poly], True, (0, 255, 0), 2)
                    anchor = tuple(poly[0])
                else:
                    anchor = (10, 30 + i * 24)

                cv2.putText(
                    debug,
                    f'QR: {text}',
                    anchor,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                is_new = text not in self._seen_texts
                if is_new or self.publish_duplicates:
                    self._instruction_pub.publish(String(data=text))
                    if is_new:
                        self._seen_texts.add(text)
                        self.get_logger().info(f'New QR instruction: {text}')

        if not found_any:
            cv2.putText(
                debug,
                'QR: none',
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        try:
            self._debug_pub.publish(self.bridge.cv2_to_imgmsg(debug, 'bgr8'))
        except CvBridgeError:
            pass


def main():
    rclpy.init(args=None)
    node = QrReaderTask2Node()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
