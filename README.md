# rins-upsilon

TurtleBot4 + ROS2 Jazzy — Task 1: face and ring detection with autonomous exploration.

## Package layout

```
upsilon/
├── upsilon/
│   ├── perception_utils.py   shared library (depth, TF, deduplication)
│   ├── face_detector.py      YOLOv8n person detection → /detected_faces
│   ├── ring_detector.py      ellipse + HSV colour detection → /detected_rings
│   ├── speech.py             espeak wrapper, subscribes /speech
│   └── controller.py         FSM mission controller (EXPLORE → APPROACH → INTERACT → DONE)
└── launch/
    ├── task1.launch.py         full system (Gazebo + RViz + SLAM + Nav2 + upsilon)
    └── detectors_only.launch.py  detectors + speech only (no navigation)
```

---

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select upsilon
source install/setup.bash
```

---

## Running the full simulation

```bash
# Blue demo world (default)
ros2 launch upsilon task1.launch.py

# Green demo world
ros2 launch upsilon task1.launch.py world:=task1_green_demo

# Yellow demo world
ros2 launch upsilon task1.launch.py world:=task1_yellow_demo

# Without RViz
ros2 launch upsilon task1.launch.py rviz:=false

# Custom spawn pose
ros2 launch upsilon task1.launch.py x:=1.0 y:=2.0 yaw:=1.57
```

This brings up Gazebo, spawns the TurtleBot4, starts SLAM + Nav2, launches RViz,
and starts all upsilon nodes.

---

## Testing detectors only (no navigation)

Useful for checking detection quality without running the full stack.

```bash
# Terminal 1 — start Gazebo + robot (reuse dis_tutorial3's launch)
ros2 launch dis_tutorial3 sim_turtlebot_slam.launch.py world:=task1_blue_demo rviz:=true

# Terminal 2 — start detectors + speech
ros2 launch upsilon detectors_only.launch.py
```

---

## Monitoring detections

```bash
# Watch face detections in map frame
ros2 topic echo /detected_faces

# Watch ring detections (colour encoded in frame_id as "map/<colour>")
ros2 topic echo /detected_rings

# Watch markers in RViz
#   /face_markers  — yellow spheres at face positions
#   /ring_markers  — coloured cylinders at ring positions
```

---

## Sending speech manually

```bash
ros2 topic pub --once /speech std_msgs/msg/String '{data: "Hello!"}'
ros2 topic pub --once /speech std_msgs/msg/String '{data: "I found a blue ring."}'
```

---

## Tuning

| What | Where |
|------|-------|
| Exploration waypoints | `controller.py` → `EXPLORATION_WAYPOINTS` |
| Approach distance to targets | `controller.py` → `APPROACH_DISTANCE` |
| Ring HSV colour ranges | `ring_detector.py` → `COLOUR_RANGES` |
| Face detection confidence | `face_detector.py` → `CONFIDENCE_THRESHOLD` |
| Deduplication merge radius | `perception_utils.py` → `IncrementalTrackManager(merge_distance=...)` |

---

## Topics

| Topic | Type | Publisher | Subscriber |
|-------|------|-----------|------------|
| `/oakd/rgb/preview/image_raw` | `sensor_msgs/Image` | Gazebo | face_detector, ring_detector |
| `/oakd/rgb/preview/depth/points` | `sensor_msgs/PointCloud2` | Gazebo | face_detector, ring_detector |
| `/detected_faces` | `geometry_msgs/PointStamped` | face_detector | controller |
| `/detected_rings` | `geometry_msgs/PointStamped` | ring_detector | controller |
| `/face_markers` | `visualization_msgs/MarkerArray` | face_detector | RViz |
| `/ring_markers` | `visualization_msgs/MarkerArray` | ring_detector | RViz |
| `/speech` | `std_msgs/String` | controller | speech |
