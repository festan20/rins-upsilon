"""Anomaly detection node — runs a UNet++ segmentation model on the latest tile image
to detect surface anomalies (cracks, damage, etc.).

Depends on tile_detection node: subscribes to /tile_detection/result for the warped
512x512 tile. Call /detect_tile first (or they can be chained).

Service:  /detect_anomaly  (std_srvs/srv/Trigger)
  Returns: success=True if anomaly detected, False if tile is okay (or on error).
           message contains the max anomaly score.

Publishes:
  /anomaly_detection/debug  (sensor_msgs/msg/Image) — tile with anomaly mask overlaid in red
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
import torch

from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from cv_bridge import CvBridge, CvBridgeError

from upsilon.anomaly_model import build_model


QOS_LATEST = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# ImageNet normalization (matches training transforms)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_SIZE = 256  # model input size


class AnomalyDetectorNode(Node):
    def __init__(self):
        super().__init__('anomaly_detector')

        self.declare_parameter('checkpoint', '')
        self.declare_parameter('encoder', 'efficientnet-b4')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('tile_topic', '/tile_detection/result')

        checkpoint   = self.get_parameter('checkpoint').get_parameter_value().string_value
        encoder      = self.get_parameter('encoder').get_parameter_value().string_value
        self.threshold = self.get_parameter('threshold').get_parameter_value().double_value
        tile_topic   = self.get_parameter('tile_topic').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.latest_tile = None
        self.model = None

        self._load_model(checkpoint, encoder)

        self.create_subscription(Image, tile_topic, self._tile_cb, QOS_LATEST)
        self.debug_pub = self.create_publisher(Image, '/anomaly_detection/debug', 10)
        self.create_service(Trigger, 'detect_anomaly', self._detect_anomaly_cb)

        self.get_logger().info(
            f'Anomaly detector ready. threshold={self.threshold}. Call /detect_anomaly.'
        )

    def _load_model(self, checkpoint: str, encoder: str) -> None:
        self.device = torch.device('cpu')
        self.model = build_model(encoder=encoder, pretrained=False).to(self.device)
        if checkpoint:
            try:
                state = torch.load(checkpoint, map_location=self.device)
                self.model.load_state_dict(state)
                self.model.eval()
                self.get_logger().info(f'Loaded checkpoint: {checkpoint}')
            except Exception as e:
                self.get_logger().error(f'Failed to load checkpoint {checkpoint}: {e}')
                self.model = None
        else:
            self.get_logger().warn('No checkpoint provided — model will not run. '
                                   'Set the "checkpoint" parameter.')
            self.model = None

    def _tile_cb(self, msg: Image) -> None:
        try:
            self.latest_tile = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError:
            pass

    def _detect_anomaly_cb(self, request, response):
        if self.model is None:
            response.success = False
            response.message = 'Model not loaded — provide a checkpoint path.'
            return response

        if self.latest_tile is None:
            response.success = False
            response.message = 'No tile image received yet. Call /detect_tile first.'
            return response

        prob_map = self._run_inference(self.latest_tile)
        max_score = float(prob_map.max())
        anomaly = max_score > self.threshold

        debug_img = self._make_debug(self.latest_tile, prob_map)
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, 'bgr8'))

        response.success = anomaly
        label = 'ANOMALY DETECTED' if anomaly else 'okay'
        response.message = f'{label} — max score: {max_score:.3f} (threshold: {self.threshold})'
        self.get_logger().info(response.message)
        return response

    @torch.no_grad()
    def _run_inference(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        normalized = (resized - MEAN) / STD
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (H, W) float32
        return prob

    def _make_debug(self, bgr: np.ndarray, prob_map: np.ndarray) -> np.ndarray:
        vis = cv2.resize(bgr, (IMAGE_SIZE, IMAGE_SIZE))
        mask = (prob_map > self.threshold).astype(np.uint8)
        overlay = vis.copy()
        overlay[mask > 0] = (
            0.6 * overlay[mask > 0] + 0.4 * np.array([0, 0, 255])
        ).astype(np.uint8)
        # score heatmap bar on top
        score_h = 16
        heatmap_row = (prob_map.mean(axis=0) * 255).astype(np.uint8)
        heatmap_row = cv2.applyColorMap(
            np.tile(heatmap_row, (score_h, 1)), cv2.COLORMAP_JET
        )
        return np.vstack([heatmap_row, overlay])


def main():
    rclpy.init()
    node = AnomalyDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
