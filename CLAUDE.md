# CLAUDE.md — ROS2 TurtleBot4 Competition Project

## Workspace Layout

```
/home/stefan/ros2_ws/src/rins-upsilon/
├── CLAUDE.md              ← this file
├── instructions/          ← planning documents (read-only reference)
│   ├── 01_deep_research.md
│   ├── 02_core_plan.md   ← PRIMARY PLAN — read this first
│   ├── 03_critique.md
│   └── 03b_user_llm_full_talk.md
├── README.md
├── dis_tutorial1/         ← REFERENCE ONLY — do not modify
├── dis_tutorial2/         ← REFERENCE ONLY — do not modify
├── dis_tutorial3/         ← REFERENCE ONLY — main nav/simulation baseline
├── dis_tutorial4/         ← REFERENCE ONLY — TF transforms patterns
├── dis_tutorial5/         ← REFERENCE ONLY — ring detection patterns
└── upsilon/               ← ACTIVE WORK — all implementation goes here
```

**Rule: only write code inside `upsilon/`. Everything else is read-only reference material.**

## Project Goal

University semester project using **TurtleBot4 + ROS2 Jazzy (Python only)**.

**Task 1** (current):
- Robot is placed in a fenced arena with **3 face posters** and **2 colored rings** on walls
- Robot must: build/use a map, explore the space, detect all faces and rings
- On face detection: approach and greet ("Hello!")
- On ring detection: approach and announce its color
- Stop when all 3 faces and 2 rings are found
- No false detections; never approach the same target twice

**Ring colors**: blue, yellow, black, green, purple, orange (2 will be present per run)

**Competition worlds** (Gazebo simulation):
- `task1_blue_demo`, `task1_green_demo`, `task1_yellow_demo`

## Architecture

Code lives under `upsilon/`. The `point_navigator/` directory contains the existing starting code.
**Note: `megatron` is just a proposed package name from the planning docs — the actual package name is TBD.**

Key nodes to implement:
| File | Purpose |
|------|---------|
| `face_detector.py` | YOLOv8 person detection on OAK-D RGB + depth → map-frame poses |
| `ring_detector.py` | HSV contour detection on OAK-D RGB + depth → map-frame poses + color |
| `controller.py` | FSM mission controller: exploration → approach → announce → resume |
| `speech.py` | Non-blocking `espeak` subprocess for speech output |
| `launch/task1.launch.py` | Full system launch |
| `launch/detectors_only.launch.py` | Detector-only launch for testing |

## Key Technical Decisions

- **Face detection**: YOLOv8 person detection — faces are printed posters on walls
- **Ring detection**: HSV segmentation + contour filtering — rings are 2D textured wall images
- **Speech**: `espeak` via subprocess (non-blocking)
- **Exploration**: Hand-coded coverage waypoints + Nav2
- **Messages**: Standard ROS2 msgs only (no custom types)
- **Deduplication**: Applied in both detectors and controller

## Camera Topics (OAK-D)

- RGB: `/oakd/rgb/preview/image_raw`
- Depth pointcloud: `/oakd/rgb/preview/depth/points`

## Reference Code Locations

- Navigation baseline: `dis_tutorial3/scripts/robot_commander.py`
- People detection pattern: `dis_tutorial3/scripts/detect_people.py`
- Ring detection pattern: `dis_tutorial3/scripts/extract_color_from_pointcloud.py`, `dis_tutorial5/scripts/detect_rings.py`
- TF transforms: `dis_tutorial4/scripts/transform_point.py`

## Open Questions

- What is the final package name (instead of `megatron`)?
- Is the map pre-built or does the robot map first each run?
- Where is `yolov8n.pt` located in the workspace?
