"""Face detector node.

Detects people (face posters) using YOLOv8n, lifts detections to 3D using
the OAK-D depth point cloud, deduplicates with IncrementalTrackManager,
and publishes unique detections + persistent markers.

Published topics
----------------
/detected_faces       (geometry_msgs/PointStamped)  — one message per NEW unique face
/face_markers         (visualization_msgs/MarkerArray) — RViz visualisation
/face_detector/debug  (sensor_msgs/Image) — annotated BGR frame with bounding boxes
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from cv_bridge import CvBridge, CvBridgeError
import cv2

from ultralytics import YOLO

from upsilon.perception_utils import TF2Helper, IncrementalTrackManager

CONFIDENCE_THRESHOLD = 0.5


class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('face_detector')

        self.declare_parameter('device', '')
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.tf2 = TF2Helper(self)
        self.tracker = IncrementalTrackManager(merge_distance=0.8)

        self.detection_color = (0, 0, 255)
        self.faces = []

        self.model = YOLO('yolov8n.pt')

        qos = qos_profile_sensor_data

        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self._rgb_cb, qos)
        self.create_subscription(PointCloud2, '/oakd/rgb/preview/depth/points', self._cloud_cb, qos)

        self._face_pub = self.create_publisher(PointStamped, '/detected_faces', 10)
        self._marker_pub = self.create_publisher(
            MarkerArray, '/face_markers', QoSReliabilityPolicy.BEST_EFFORT
        )
        self._debug_pub = self.create_publisher(Image, '/face_detector/debug', 10)

        self.get_logger().info('Face detector ready.')

    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image) -> None:
        self.faces = []

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridgeError: {e}')
            return

        try:
            res = self.model.predict(
                cv_image, imgsz=(256, 320), show=False, verbose=False,
                classes=[0], device=self.device, conf=CONFIDENCE_THRESHOLD,
            )

            for x in res:
                bbox = x.boxes.xyxy
                if bbox.nelement() == 0:
                    continue

                for i in range(len(bbox)):
                    b = bbox[i]
                    x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), self.detection_color, 3)
                    cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)

                    conf = float(x.boxes.conf[i])
                    cv2.putText(cv_image, f'{conf:.2f}', (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.detection_color, 1)

                    self.faces.append((cx, cy))
        except Exception as e:
            self.get_logger().error(f'YOLO error: {e}', throttle_duration_sec=5.0)
            cv2.putText(cv_image, f'YOLO error: {e}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Status text — always drawn, always published
        n_tracked = self.tracker.track_count
        cv2.putText(cv_image, f'det:{len(self.faces)} tracked:{n_tracked}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        try:
            self._debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        except CvBridgeError:
            pass

    # ------------------------------------------------------------------
    def _cloud_cb(self, msg: PointCloud2) -> None:
        if not self.faces:
            return

        height = msg.height
        width = msg.width

        if height == 0 or width == 0:
            return

        a = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'))
        a = a.reshape((height, width, 3))

        for cx, cy in self.faces:
            if cy < 0 or cy >= height or cx < 0 or cx >= width:
                continue

            d = a[cy, cx, :]

            if not np.isfinite(d).all() or d[2] <= 0.0:
                continue

            # Build PointStamped in camera frame, transform to map
            ps = PointStamped()
            ps.header.frame_id = msg.header.frame_id
            ps.header.stamp = msg.header.stamp
            ps.point.x = float(d[0])
            ps.point.y = float(d[1])
            ps.point.z = float(d[2])

            ps_map = self.tf2.transform_point(ps, 'map')
            if ps_map is None:
                continue

            mx, my = ps_map.point.x, ps_map.point.y
            track_id, is_new = self.tracker.update(mx, my)

            if is_new:
                self.get_logger().info(
                    f'New face #{track_id} at map ({mx:.2f}, {my:.2f})'
                )
                self._face_pub.publish(ps_map)

        self._publish_markers()
        self.faces = []

    # ------------------------------------------------------------------
    def _publish_markers(self) -> None:
        arr = MarkerArray()
        for track in self.tracker._tracks:
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'faces'
            m.id = track['id']
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = track['x']
            m.pose.position.y = track['y']
            m.pose.position.z = 1.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=1.0)
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def main():
    rclpy.init()
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
