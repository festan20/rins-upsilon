# Task 1 Plan — Face & Ring Detection System

## Overview

This plan covers the remaining phases of the Task 1 roadmap: detecting face posters, detecting colored rings, orchestrating mission behavior, adding speech output, wiring launch files, and validating the full system in simulation.

The implementation should live primarily in the `megatron` package. The `vendor/` tree contains the ROS 2 documentation and the lab tutorial repositories (used for reference material), with `dis_tutorial3` still serving as the main navigation and simulation baseline and `dis_tutorial4` / `dis_tutorial5` providing additional TF and ring-detection patterns.

## Core architectural idea

The Gazebo assets are not real people or 3D rings. The “faces” are photo posters on walls, and the “rings” are 2D textured ring images mounted on walls. That means the problem is mostly computer vision from the OAK-D camera feed, plus depth lookup and TF transforms into `map` coordinates.

## Phase 1 — Face detection node

### Goal
Detect up to 3 face posters, estimate their 3D positions in the map frame, and publish usable detections for the controller.

### Implementation plan
1. Add a detector abstraction so the face detector can be swapped later without rewriting the node.
2. Implement `FaceDetectorNode` in `src/megatron/megatron/face_detector.py`.
3. Subscribe to:
   - `/oakd/rgb/preview/image_raw`
   - `/oakd/rgb/preview/depth/points`
4. On each RGB image:
   - run the detector,
   - collect bounding boxes,
   - keep the detection center for each candidate.
5. On each point cloud:
   - read the 3D point at the detection center,
   - transform the point from the camera frame into `map` using TF2.
6. Confirm detections across multiple frames before publishing.
7. Deduplicate nearby detections so the same poster is not reported repeatedly.
8. Publish:
   - `visualization_msgs/Marker` on `/face_markers`
   - `geometry_msgs/PoseStamped` on `/detected_faces`
   - optionally an annotated image for debugging/demo purposes.

### Solutions considered
- **YOLOv8 person detection**: easiest because the repository already includes `yolov8n.pt` and the existing tutorial code uses the same general pattern.
- **OpenCV DNN face detector**: could be more semantically correct for faces, but would require more setup and may not match the cropped poster appearance well.
- **A face-specific YOLO model**: potentially stronger accuracy, but it adds extra model-management complexity.

### Subjective pick
Start with **YOLOv8 person detection** behind a detector abstraction. It is the fastest path to a working system and keeps the door open for a model swap if the posters are not detected reliably.

### Relevant references
- `vendor/dis_tutorial3/scripts/detect_people.py`
- `yolov8n.pt`

## Phase 2 — Ring detection node

### Goal
Detect up to 2 colored rings and determine each ring’s color.

### Implementation plan
1. Implement `RingDetectorNode` in `src/megatron/megatron/ring_detector.py`.
2. Subscribe to the same RGB and depth topics as the face detector.
3. For each frame:
   - convert BGR to HSV,
   - threshold for each ring color,
   - find contours,
   - filter by area and circularity,
   - optionally verify the ring structure with hole/child-contour logic.
4. Extract the centroid of each accepted contour.
5. Look up the corresponding 3D point from the depth cloud.
6. Transform the result into `map`.
7. Apply the same multi-frame confirmation and deduplication strategy used for faces.
8. Publish:
   - `visualization_msgs/Marker` on `/ring_markers`
   - `geometry_msgs/PoseStamped` on `/detected_rings`
   - `std_msgs/String` on `/detected_ring_color`
   - optionally an annotated image for debugging/demo purposes.

### Solutions considered
- **HSV + contour segmentation**: simple, fast, and well matched to solid-colored wall textures.
- **Template matching**: less robust to scale and lighting changes.
- **Color histogram classification on cropped regions**: viable, but overkill for the fixed Gazebo assets.

### Subjective pick
Use **HSV segmentation plus contour filtering**. It is the simplest method that still matches the task geometry well and should be easy to tune in the simulator.

### Relevant references
- `vendor/dis_tutorial3/scripts/extract_color_from_pointcloud.py`
- `vendor/dis_tutorial5/scripts/detect_rings.py`

## Phase 3 — Mission controller

### Goal
Coordinate exploration, react to detections, approach targets, greet or announce them, and stop when all required objects are found.

### Implementation plan
1. Create `src/megatron/megatron/controller.py`.
2. Build a `MissionController` node that manages the mission state machine.
3. Reuse or embed the relevant `RobotCommander` behavior from `dis_tutorial3`.
4. Subscribe to:
   - `/detected_faces`
   - `/detected_rings`
   - `/detected_ring_color`
5. Maintain internal state for:
   - confirmed faces,
   - confirmed rings with colors,
   - current waypoint index,
   - current mission state.
6. When a new detection arrives:
   - reject it if it is too close to a previously confirmed target,
   - compute an approach pose,
   - pause exploration,
   - send the navigation goal.
7. After reaching the target:
   - speak a greeting for a face,
   - speak a color announcement for a ring,
   - resume exploration.
8. When all 3 faces and 2 rings are found:
   - stop exploration,
   - cancel any active goal,
   - log mission completion.

### Solutions considered
- **Simple callback-driven controller**: easy to write, but can become messy once exploration and interruption handling overlap.
- **Finite-state machine**: a little more upfront structure, but much cleaner for exploration/approach/announce transitions.
- **Behavior tree**: powerful, but too heavy for this scope.

### Subjective pick
Use a **finite-state machine** inside the controller. It is the best balance of simplicity and reliability for this task.

### Relevant references
- `vendor/dis_tutorial3/scripts/robot_commander.py`
- `vendor/dis_tutorial4/scripts/transform_point.py`

## Phase 4 — Speech synthesis

### Goal
Provide lightweight, non-blocking spoken feedback for greetings and ring color announcements.

### Implementation plan
1. Create `src/megatron/megatron/speech.py`.
2. Implement a small `Speaker` utility class with a `speak(text: str)` method.
3. Use `espeak` through a subprocess call so speech does not block the controller.
4. Keep `pyttsx3` as a fallback if needed.
5. Call the utility from the controller instead of creating a separate ROS node.

### Solutions considered
- **`espeak` subprocess**: minimal dependencies and simple non-blocking behavior.
- **`pyttsx3`**: convenient fallback, but can be heavier and less predictable.
- **ROS audio pipeline / custom node**: more complex than needed for this task.

### Subjective pick
Use **`espeak` subprocess** as the primary path. It is the lightest and most reliable option for this environment.

## Phase 5 — Launch and integration

### Goal
Make the package easy to run and test end to end.

### Implementation plan
1. Create a launch directory under `megatron` if it does not already exist.
2. Add `task1.launch.py` to start the simulation, detectors, and controller together.
3. Add `detectors_only.launch.py` for focused detector testing.
4. Update `setup.py` so launch files and other package assets install correctly.
5. Update `package.xml` with the needed runtime dependencies.
6. Optionally add an RViz config that shows the map, robot, scan data, camera feed, and detection markers.

### Relevant files
- `src/megatron/launch/task1.launch.py`
- `src/megatron/launch/detectors_only.launch.py`
- `src/megatron/config/rviz.rviz`
- `src/megatron/setup.py`
- `src/megatron/package.xml`

## Phase 6 — Exploration strategy

### Goal
Cover the arena efficiently so the robot can discover all posters and rings.

### Implementation plan
1. Hand-code a set of waypoints that covers the arena well.
2. Cycle through waypoints one at a time using Nav2.
3. Keep the detectors running continuously during navigation.
4. Optionally cancel the current goal if a promising detection appears.
5. Optionally spin in place at some waypoints to improve camera coverage.

### Solutions considered
- **Simple perimeter loop**: easiest to implement, but may miss wall-mounted targets in the middle.
- **Zigzag / coverage path**: better coverage of the whole arena.
- **Reactive wandering**: interesting, but less deterministic and harder to debug.

### Subjective pick
Use a **hand-coded coverage path** with optional spins. That gives the most predictable behavior while still being easy to tune.

## Phase 7 — Testing and tuning

### Goal
Make the whole system reliable across the available simulation worlds.

### Test plan
1. Run the system in all 3 worlds:
   - `task1_blue_demo`
   - `task1_green_demo`
   - `task1_yellow_demo`
2. Tune HSV thresholds for ring colors.
3. Tune the detector confidence threshold for faces.
4. Tune multi-frame confirmation and deduplication distance thresholds.
5. Adjust waypoints if needed to reduce search time.
6. Validate that the mission completes without duplicate detections.
7. Validate that false detections stay low.

### Verification checklist
- `colcon build --packages-select megatron` succeeds.
- The simulation launches correctly.
- The face detector publishes stable markers.
- The ring detector publishes stable markers and correct color labels.
- The full launch runs the full mission autonomously.
- No duplicate markers pile up at the same location.
- No markers appear at obviously incorrect locations.

## Overall design decisions

- Use **YOLOv8 person detection** for faces first.
- Use **HSV + contour filtering** for rings.
- Use **`espeak` subprocess** for speech output.
- Avoid custom message types and stick to standard ROS 2 messages.
- Apply **deduplication in both the detectors and the controller**.
- Keep all new logic in the **`megatron` package**.
- Use **hand-coded waypoints** for exploration.

## File summary

| File | Action |
| --- | --- |
| `src/megatron/megatron/face_detector.py` | Implement or rewrite |
| `src/megatron/megatron/ring_detector.py` | Implement or rewrite |
| `src/megatron/megatron/controller.py` | Create |
| `src/megatron/megatron/speech.py` | Create |
| `src/megatron/launch/task1.launch.py` | Create |
| `src/megatron/launch/detectors_only.launch.py` | Create |
| `src/megatron/setup.py` | Update |
| `src/megatron/package.xml` | Update |
| `src/megatron/config/rviz.rviz` | Optional create |

## Notes and caveats

1. If YOLOv8 person detection does not reliably detect the face posters, switch the detector implementation while keeping the same interface.
2. Approaching a face or ring requires a reasonable approximation of “in front of the object”; the controller can use the detection geometry and the robot’s relative position to compute this.
3. Detections may arrive while the robot is already navigating, so the controller must handle interruptions cleanly.
4. The plan is intentionally incremental: get each detector working alone before wiring in the full mission logic.
