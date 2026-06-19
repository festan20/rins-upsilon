#!/usr/bin/env python3
"""Generate a PDF report of detected rings and barrels.

Subscribes to /ring_markers_task2 and /cylinder_markers_task2, collects data
for COLLECT_DURATION seconds, filters by threshold, then writes a PDF to
OUTPUT_DIR.

Run with:
    ros2 run upsilon report
"""
import os
import re
import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from visualization_msgs.msg import MarkerArray

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    )
except ImportError:
    raise SystemExit(
        'reportlab is not installed. Run: pip install reportlab'
    )

# ---------------------------------------------------------------------------
# Thresholds — edit these to change which detections appear in the report
# ---------------------------------------------------------------------------
RING_THRESHOLD = 5     # min detection count for a ring to be reported
BARREL_THRESHOLD = 4     # min detection count for a barrel to be reported

OUTPUT_DIR = os.path.join(
    os.path.expanduser('~'), 'colcon_ws', 'rins-upsilon', 'upsilon', 'upsilon', 'reports'
)
# ---------------------------------------------------------------------------


def _parse_ring_label(text: str):
    """'{colour} (n={count})' → (colour, count) or (None, 0)"""
    m = re.match(r'(\w+)\s+\(n=(\d+)\)', text)
    if m:
        return m.group(1), int(m.group(2))
    return None, 0


def _parse_barrel_label(text: str):
    """'{colour} {orientation}[ LEAKING] (n={count})' → (colour, orientation, leaking, count)"""
    m = re.match(r'(\w+)\s+(upright|fallen)( LEAKING)?\s+\(n=(\d+)\)', text)
    if m:
        return m.group(1), m.group(2), (m.group(3) is not None), int(m.group(4))
    return None, 'unknown', False, 0


class ReportCollector(Node):
    def __init__(self):
        super().__init__('report_collector')
        self._ring_data: dict[int, tuple[str, int]] = {}
        self._barrel_data: dict[int, tuple[str, str, bool, int]] = {}

        self.create_subscription(MarkerArray, '/ring_markers_task2',     self._ring_cb,   10)
        self.create_subscription(MarkerArray, '/cylinder_markers_task2', self._barrel_cb, 10)
        self.create_subscription(Bool, '/generate_report', self._trigger_cb, 10)
        self.get_logger().info(
            f'Report node ready — listening for markers '
            f'(ring≥{RING_THRESHOLD}, barrel≥{BARREL_THRESHOLD}). '
            f'Publish to /generate_report to write PDF.'
        )

    def _ring_cb(self, msg: MarkerArray) -> None:
        for m in msg.markers:
            if m.ns == 'ring_labels_task2':
                colour, count = _parse_ring_label(m.text)
                if colour:
                    self._ring_data[m.id] = (colour, count)

    def _barrel_cb(self, msg: MarkerArray) -> None:
        for m in msg.markers:
            if m.ns == 'cylinder_labels_task2':
                colour, orientation, leaking, count = _parse_barrel_label(m.text)
                if colour:
                    self._barrel_data[m.id] = (colour, orientation, leaking, count)

    def rings_above_threshold(self):
        return [(c, n) for c, n in self._ring_data.values() if n >= RING_THRESHOLD]

    def barrels_above_threshold(self):
        return [(c, o, l, n) for c, o, l, n in self._barrel_data.values()
                if n >= BARREL_THRESHOLD]

    def _trigger_cb(self, msg: Bool) -> None:
        if not msg.data:
            return
        rings   = self.rings_above_threshold()
        barrels = self.barrels_above_threshold()
        self.get_logger().info(
            f'Report triggered — {len(rings)} ring(s), {len(barrels)} barrel(s) above threshold.'
        )
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(OUTPUT_DIR, f'report_{timestamp}.pdf')
        generate_pdf(rings, barrels, output_path)
        self.get_logger().info(f'Report saved → {output_path}')


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _make_table(data, col_widths):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  colors.HexColor('#444444')),
        ('TEXTCOLOR',     (0, 0), (-1, 0),  colors.white),
        ('FONTNAME',      (0, 0), (-1, 0),  'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
        ('GRID',          (0, 0), (-1, -1),  0.5, colors.HexColor('#cccccc')),
        ('ALIGN',         (1, 0), (-1, -1),  'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1),  'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, -1),  4),
        ('BOTTOMPADDING', (0, 0), (-1, -1),  4),
    ]))
    return t


def generate_pdf(rings, barrels, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    story.append(Paragraph('Detection Report — Task 2', styles['Title']))
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    story.append(Paragraph(f'Generated: {ts}', styles['Normal']))
    story.append(Paragraph(
        f'Ring threshold: n ≥ {RING_THRESHOLD} &nbsp;&nbsp; '
        f'Barrel threshold: n ≥ {BARREL_THRESHOLD}',
        styles['Normal'],
    ))
    story.append(Spacer(1, 0.6 * cm))

    # -----------------------------------------------------------------------
    # Rings
    # -----------------------------------------------------------------------
    story.append(Paragraph(f'Rings (n ≥ {RING_THRESHOLD})', styles['Heading1']))

    if rings:
        by_colour: dict[str, int] = {}
        for colour, _ in rings:
            by_colour[colour] = by_colour.get(colour, 0) + 1

        story.append(Paragraph(f'Total rings: <b>{len(rings)}</b>', styles['Normal']))
        story.append(Spacer(1, 0.25 * cm))

        data = [['Colour', 'Count']]
        for colour in sorted(by_colour):
            data.append([colour.capitalize(), str(by_colour[colour])])

        story.append(_make_table(data, [7*cm, 7*cm]))
    else:
        story.append(Paragraph(
            f'No rings met the threshold (n ≥ {RING_THRESHOLD}).', styles['Normal']))

    story.append(Spacer(1, 0.8 * cm))

    # -----------------------------------------------------------------------
    # Barrels
    # -----------------------------------------------------------------------
    story.append(Paragraph(f'Barrels (n ≥ {BARREL_THRESHOLD})', styles['Heading1']))

    if barrels:
        total_upright = sum(1 for _, o, _, _ in barrels if o == 'upright')
        total_fallen  = sum(1 for _, o, _, _ in barrels if o == 'fallen')
        total_leaking = sum(1 for _, _, l, _ in barrels if l)

        story.append(Paragraph(
            f'Total barrels: <b>{len(barrels)}</b> &nbsp;—&nbsp; '
            f'Upright: <b>{total_upright}</b> &nbsp; '
            f'Fallen: <b>{total_fallen}</b> &nbsp; '
            f'Leaking: <b>{total_leaking}</b>',
            styles['Normal'],
        ))
        story.append(Spacer(1, 0.25 * cm))

        data = [['Colour', 'Orientation', 'Leaking']]
        for colour, orientation, leaking, _ in sorted(barrels, key=lambda b: (b[0], b[1])):
            data.append([colour.capitalize(), orientation.capitalize(), 'Yes' if leaking else 'No'])

        story.append(_make_table(data, [5*cm, 5*cm, 4*cm]))
    else:
        story.append(Paragraph(
            f'No barrels met the threshold (n ≥ {BARREL_THRESHOLD}).', styles['Normal']))

    doc.build(story)
    print(f'Report saved → {output_path}')


# ---------------------------------------------------------------------------

def main():
    rclpy.init()
    node = ReportCollector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
