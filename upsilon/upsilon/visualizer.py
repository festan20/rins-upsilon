"""Visualizer node — opens 4 OpenCV windows.

Windows:
  1. Camera POV          — raw RGB from OAK-D
  2. Depth View          — colourised depth from PointCloud2
  3. Face Detection View — annotated debug image from face_detector
  4. Ring Detection View — annotated debug image from ring_detector
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError

WINDOW_W = 640
WINDOW_H = 480


class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer')
        self.bridge = CvBridge()

        qos = qos_profile_sensor_data

        # Raw camera
        self.create_subscription(
            Image, '/oakd/rgb/preview/image_raw', self._rgb_cb, qos)
        # Depth pointcloud
        self.create_subscription(
            PointCloud2, '/oakd/rgb/preview/depth/points', self._cloud_cb, qos)
        # Face detector debug
        self.create_subscription(
            Image, '/face_detector/debug', self._face_debug_cb, qos)
        # Ring detector debug
        self.create_subscription(
            Image, '/ring_detector/debug', self._ring_debug_cb, qos)

        # Create named windows
        cv2.namedWindow('Camera POV', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Depth View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Face Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ring Detection', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('Camera POV', WINDOW_W, WINDOW_H)
        cv2.resizeWindow('Depth View', WINDOW_W, WINDOW_H)
        cv2.resizeWindow('Face Detection', WINDOW_W, WINDOW_H)
        cv2.resizeWindow('Ring Detection', WINDOW_W, WINDOW_H)

        # Periodic waitKey so OpenCV processes GUI events
        self.create_timer(0.03, self._gui_tick)

        self.get_logger().info('Visualizer ready — 4 windows opened.')

    def _rgb_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return
        cv2.imshow('Camera POV', frame)

    def _cloud_cb(self, msg: PointCloud2) -> None:
        w = msg.width
        h = msg.height
        if w == 0 or h == 0:
            return

        point_step = msg.point_step

        # Reshape raw bytes into (h*w, point_step) and extract z as float32
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        points = raw.reshape(h * w, point_step)
        # z is at byte offset 8, 4 bytes float32
        depth = points[:, 8:12].copy().view(np.float32).reshape(h, w)
        # Mark invalid values as 0
        depth[~np.isfinite(depth) | (depth <= 0)] = 0.0

        # Normalise to 0-255 for visualisation
        valid = depth[depth > 0]
        if valid.size == 0:
            return
        max_d = np.percentile(valid, 95)
        depth_clipped = np.clip(depth, 0, max_d)
        depth_norm = (depth_clipped / max_d * 255).astype(np.uint8)
        # Invert so closer = brighter
        depth_norm = 255 - depth_norm
        # Zero-depth pixels (invalid) → black
        depth_norm[depth == 0] = 0

        coloured = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        coloured[depth == 0] = [0, 0, 0]

        cv2.imshow('Depth View', coloured)

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
