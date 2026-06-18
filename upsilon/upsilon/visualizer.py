"""Visualizer node — single tiled window showing all debug feeds.

All active image streams are composited into one resizable OpenCV window
arranged in a grid (up to N_COLS columns). Panels without data yet show a
black placeholder with the stream name.
"""

import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

CELL_W  = 320
CELL_H  = 240
N_COLS  = 3

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
                ('qr_debug_topic', ''),
                ('ring_debug_topic', '/ring_detector/debug'),
                ('cylinder_debug_topic', ''),
                ('tile_result_topic', '/tile_detection/result'),
                ('tile_debug_topic', '/tile_detection/debug'),
                ('anomaly_debug_topic', '/anomaly_detection/debug'),
            ],
        )
        self.rgb_topic            = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.compressed_rgb       = self.get_parameter('compressed_rgb').get_parameter_value().bool_value
        self.top_rgb_topic        = self.get_parameter('top_rgb_topic').get_parameter_value().string_value
        self.top_compressed_rgb   = self.get_parameter('top_compressed_rgb').get_parameter_value().bool_value
        self.face_debug_topic     = self.get_parameter('face_debug_topic').get_parameter_value().string_value
        self.qr_debug_topic       = self.get_parameter('qr_debug_topic').get_parameter_value().string_value
        self.ring_debug_topic     = self.get_parameter('ring_debug_topic').get_parameter_value().string_value
        self.cylinder_debug_topic = self.get_parameter('cylinder_debug_topic').get_parameter_value().string_value
        self.tile_result_topic    = self.get_parameter('tile_result_topic').get_parameter_value().string_value
        self.tile_debug_topic     = self.get_parameter('tile_debug_topic').get_parameter_value().string_value
        self.anomaly_debug_topic  = self.get_parameter('anomaly_debug_topic').get_parameter_value().string_value
        self.bridge = CvBridge()

        # Ordered panel list — every entry gets a cell in the grid
        self._labels: list[str] = []
        self._frames: dict[str, np.ndarray | None] = {}

        def _add(label: str) -> None:
            self._labels.append(label)
            self._frames[label] = None

        _add('Camera POV')
        _add('Top Camera POV')
        _add('Face Detection')
        _add('Ring Detection')

        rgb_msg_type     = CompressedImage if self.compressed_rgb     else Image
        top_rgb_msg_type = CompressedImage if self.top_compressed_rgb else Image

        self.create_subscription(rgb_msg_type,     self.rgb_topic,        self._rgb_cb,     QOS_LATEST)
        self.create_subscription(top_rgb_msg_type, self.top_rgb_topic,    self._top_rgb_cb, QOS_LATEST)
        self.create_subscription(Image,            self.face_debug_topic, self._face_cb,    QOS_LATEST)
        self.create_subscription(Image,            self.ring_debug_topic, self._ring_cb,    QOS_LATEST)

        if self.qr_debug_topic:
            _add('QR Detection')
            self.create_subscription(Image, self.qr_debug_topic, self._qr_cb, QOS_LATEST)
        if self.cylinder_debug_topic:
            _add('Cylinder Detection')
            self.create_subscription(Image, self.cylinder_debug_topic, self._cylinder_cb, QOS_LATEST)
        if self.tile_result_topic:
            _add('Tile Result')
            self.create_subscription(Image, self.tile_result_topic, self._tile_result_cb, QOS_LATEST)
        if self.tile_debug_topic:
            _add('Tile Debug')
            self.create_subscription(Image, self.tile_debug_topic, self._tile_debug_cb, QOS_LATEST)
        if self.anomaly_debug_topic:
            _add('Anomaly Detection')
            self.create_subscription(Image, self.anomaly_debug_topic, self._anomaly_cb, QOS_LATEST)

        cv2.namedWindow('Upsilon Vision', cv2.WINDOW_NORMAL)

        self.create_timer(0.03, self._gui_tick)
        self.get_logger().info(f'Visualizer ready — {len(self._labels)} panels in one window.')

    # ------------------------------------------------------------------
    # Image callbacks — just store the latest frame
    # ------------------------------------------------------------------
    def _decode(self, msg: Image | CompressedImage) -> np.ndarray | None:
        try:
            if isinstance(msg, CompressedImage):
                return self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            return self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            return None

    def _rgb_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Camera POV'] = f

    def _top_rgb_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Top Camera POV'] = f

    def _face_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Face Detection'] = f

    def _ring_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Ring Detection'] = f

    def _qr_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['QR Detection'] = f

    def _cylinder_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Cylinder Detection'] = f

    def _tile_result_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Tile Result'] = f

    def _tile_debug_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Tile Debug'] = f

    def _anomaly_cb(self, msg):
        f = self._decode(msg)
        if f is not None:
            self._frames['Anomaly Detection'] = f

    # ------------------------------------------------------------------
    # GUI tick — build grid and show
    # ------------------------------------------------------------------
    def _gui_tick(self) -> None:
        cells = []
        for label in self._labels:
            frame = self._frames[label]
            if frame is None:
                cell = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
            else:
                cell = cv2.resize(frame, (CELL_W, CELL_H))
            cv2.putText(cell, label, (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(cell)

        n      = len(cells)
        n_cols = min(N_COLS, n)
        n_rows = math.ceil(n / n_cols)

        blank = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        while len(cells) < n_rows * n_cols:
            cells.append(blank)

        grid = np.vstack([
            np.hstack(cells[r * n_cols:(r + 1) * n_cols])
            for r in range(n_rows)
        ])

        cv2.imshow('Upsilon Vision', grid)
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
