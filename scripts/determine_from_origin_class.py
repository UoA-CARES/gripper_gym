import cv2
import numpy as np
import pyrealsense2 as rs
from cares_lib.vision.STagDetector import STagDetector
from ultralytics import YOLO

class CubeGripperTracker:
    def __init__(self, target_id=0, marker_size=0.048, yolo_model_path="Arm.pt", realsense_id=0, window_name="Camera"):
        # Parameters
        self.TARGET_ID = target_id
        self.MARKER_SIZE = marker_size
        self.realsense_id = realsense_id
        self.window_name = window_name

        # RealSense setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        config.enable_device(str(self.realsense_id))
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        self.camera_matrix = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        self.camera_distortion = np.array(intr.coeffs[:5])

        # Detection
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.sTagDetector = STagDetector(40, 11)
        self.yolo_model = YOLO(yolo_model_path)

        #Yolo thresholding
        self.yolo_initial_conf = 0.8   # starting confidence
        self.yolo_min_conf     = 0.4   # lowest confidence
        self.yolo_conf_step    = 0.2  # how much to lower each itteration
        self.yolo_passes       = 3     # number of frames to scan

        # State
        self.last_origin = None
        self.last_axes = None
        self.last_cube_pos = None
        self.last_gripper_positions = {0: None, 1: None}

        while self.last_origin is None:
            self.update()
            print(f"Looking for origin for {self.realsense_id}")


    def _get_3d_point(self, u, v, depth):
        return np.array([(u - self.cx) * depth / self.fx,
                         (v - self.cy) * depth / self.fy,
                         depth])

    def _update_reference_frame(self, frame, depth_frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None and self.TARGET_ID in ids:
            idx = np.where(ids.flatten() == self.TARGET_ID)[0][0]
            pts = corners[idx][0]
            c0, c1, c3 = pts[0], pts[1], pts[3]
            d0 = depth_frame.get_distance(int(c0[0]), int(c0[1]))
            d1 = depth_frame.get_distance(int(c1[0]), int(c1[1]))
            d3 = depth_frame.get_distance(int(c3[0]), int(c3[1]))

            if d0 > 0 and d1 > 0 and d3 > 0:
                p0 = self._get_3d_point(c0[0], c0[1], d0)
                p1 = self._get_3d_point(c1[0], c1[1], d1)
                p3 = self._get_3d_point(c3[0], c3[1], d3)

                z_axis = p1 - p0
                x_axis = p3 - p0
                z_axis /= np.linalg.norm(z_axis)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)

                self.last_origin = p0
                self.last_axes = (x_axis, y_axis, z_axis)

    def _update_cube_position(self, frame):
        marker_poses = self.sTagDetector.get_marker_poses(frame, self.camera_matrix, self.camera_distortion, display=False)
        marker_offsets = {i: np.array([0, 0, 0.025]) for i in range(1, 7)}
        cube_centers = []

        for marker_id, pose in marker_poses.items():
            rvec = np.array(pose["r_vec"], dtype=np.float32).reshape((3, 1))
            tvec = np.array(pose["position"], dtype=np.float32).reshape((3, 1)) / 1000.0
            if marker_id in marker_offsets:
                rot_mat, _ = cv2.Rodrigues(rvec)
                offset = marker_offsets[marker_id].reshape((3, 1))
                center = tvec - rot_mat @ offset
                cube_centers.append(center)

        if cube_centers:
            self.last_cube_pos = np.mean(cube_centers, axis=0).flatten()

    def _detect_grippers_with_retries(self, passes=None, initial_conf=None, min_conf=None, conf_step=None):
        """
        Detect grippers using YOLO keypoints.
        - Retries over multiple frames to stabilize.
        - Gradually lowers confidence if needed.
        - Returns {0: 3D_point or (0,0,0), 1: 3D_point or (0,0,0)}.
        """
        if passes is None:      passes = self.yolo_passes
        if initial_conf is None: initial_conf = self.yolo_initial_conf
        if min_conf is None:     min_conf = self.yolo_min_conf
        if conf_step is None:    conf_step = self.yolo_conf_step

        conf = initial_conf
        found = {0: None, 1: None}

        target_classes = [0, 1]  # gripper class IDs

        for _ in range(passes):
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                conf = max(min_conf, conf - conf_step)
                continue

            frame = np.asanyarray(color_frame.get_data())
            results = self.yolo_model(frame, conf=conf, imgsz=960)

            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                conf = max(min_conf, conf - conf_step)
                continue

            boxes = results[0].boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            keypoints_all = getattr(results[0], "keypoints", None)

            # skip if keypoints missing
            if keypoints_all is None:
                conf = max(min_conf, conf - conf_step)
                continue

            keypoints_all = keypoints_all.xy.cpu().numpy()
            confs_all = results[0].keypoints.conf.cpu().numpy()

            # iterate detections
            for i_det, cid in enumerate(class_ids):
                if cid not in target_classes or found[cid] is not None:
                    continue

                kp_set = keypoints_all[i_det]
                for kp in kp_set:
                    u, v = int(kp[0]), int(kp[1])
                    depth = depth_frame.get_distance(u, v)
                    if depth > 0:
                        found[cid] = self._get_3d_point(u, v, depth)
                        break  # first valid keypoint

            # stop early if both grippers found
            if all(found[cid] is not None for cid in target_classes):
                break

            # lower confidence for next pass
            conf = max(min_conf, conf - conf_step)

        # fallback if not found
        for cid in target_classes:
            if found[cid] is None:
                found[cid] = (0.0, 0.0, 0.0)
            else:
                self.last_gripper_positions[cid] = found[cid]

        return found


    def _update_gripper_positions(self):
        # ignore incoming frame/depth_frame; we re-grab internally for settling
        _ = self._detect_grippers_with_retries()


    #update with visuals
    def update(self):
        """Capture and update all detections from the current frame with visual output"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return  # keep same behavior

        frame = np.asanyarray(color_frame.get_data())
        visual_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- ArUco Detection & Frame Setup ---
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None and self.TARGET_ID in ids:
            idx = np.where(ids.flatten() == self.TARGET_ID)[0][0]
            pts = corners[idx][0]
            c0, c1, c3 = pts[0], pts[1], pts[3]

            d0 = depth_frame.get_distance(int(c0[0]), int(c0[1]))
            d1 = depth_frame.get_distance(int(c1[0]), int(c1[1]))
            d3 = depth_frame.get_distance(int(c3[0]), int(c3[1]))

            if d0 > 0 and d1 > 0 and d3 > 0:
                p0 = self._get_3d_point(c0[0], c0[1], d0)
                p1 = self._get_3d_point(c1[0], c1[1], d1)
                p3 = self._get_3d_point(c3[0], c3[1], d3)

                z_axis = p1 - p0
                x_axis = p3 - p0
                z_axis /= np.linalg.norm(z_axis)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)

                self.last_origin = p0
                self.last_axes = (x_axis, y_axis, z_axis)

        # --- STag Marker Detection ---
        for i in range(5):
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return

            frame = np.asanyarray(color_frame.get_data())
            visual_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            marker_poses = self.sTagDetector.get_marker_poses(
                frame, self.camera_matrix, self.camera_distortion, display=False
            )
            marker_offsets = {i: np.array([0, 0, 0.025]) for i in range(1, 7)}
            cube_centers = []

            for marker_id, pose in marker_poses.items():
                rvec = np.array(pose["r_vec"], dtype=np.float32).reshape((3, 1))
                tvec = np.array(pose["position"], dtype=np.float32).reshape((3, 1)) / 1000.0
                cv2.drawFrameAxes(visual_frame, self.camera_matrix, self.camera_distortion, rvec, tvec, 0.05)

                if marker_id in marker_offsets:
                    rot_mat, _ = cv2.Rodrigues(rvec)
                    offset = marker_offsets[marker_id].reshape((3, 1))
                    center = tvec - rot_mat @ offset
                    cube_centers.append(center)

            if cube_centers:
                self.last_cube_pos = np.mean(cube_centers, axis=0).flatten()
                break
            else:
                self.last_cube_pos = None

        # --- YOLO Gripper Detection ---
        results = self.yolo_model(frame, conf=0.5)
        raw_positions = self._detect_grippers_with_retries()
        # raw_positions = self._update_gripper_positions()

        # Agent truth (always (0,0,0) if not found)
        self.agent_gripper_positions = {}
        for cid in (0, 1):
            if raw_positions[cid] is not None and raw_positions[cid][2] != 0:
                self.last_gripper_positions[cid] = raw_positions[cid]
                self.agent_gripper_positions[cid] = raw_positions[cid]
            else:
                self.agent_gripper_positions[cid] = (0.0, 0.0, 0.0)

        # Visualization-only fallback
        gripper_positions = {}
        for cid in (0, 1):
            if raw_positions[cid] is not None:
                gripper_positions[cid] = raw_positions[cid]
            else:
                gripper_positions[cid] = self.last_gripper_positions[cid]

        # --- Visual Debugging ---
        if self.last_origin is not None and self.last_axes is not None:
            origin_2d, _ = cv2.projectPoints(
                self.last_origin[np.newaxis], np.zeros(3), np.zeros(3),
                self.camera_matrix, self.camera_distortion
            )
            origin_2d = tuple(origin_2d[0][0].astype(int))
            cv2.circle(visual_frame, origin_2d, 6, (0, 255, 0), -1)
            cv2.putText(visual_frame, "Origin", (origin_2d[0] + 10, origin_2d[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for vec, color in zip(self.last_axes, [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                endpoint = self.last_origin + vec * 0.05
                end_2d, _ = cv2.projectPoints(
                    endpoint[np.newaxis], np.zeros(3), np.zeros(3),
                    self.camera_matrix, self.camera_distortion
                )
                end_2d = tuple(end_2d[0][0].astype(int))
                cv2.line(visual_frame, origin_2d, end_2d, color, 2)

        if self.last_cube_pos is not None and self.last_cube_pos[2] != 0:
            u = int(self.last_cube_pos[0] * self.fx / self.last_cube_pos[2] + self.cx)
            v = int(self.last_cube_pos[1] * self.fy / self.last_cube_pos[2] + self.cy)
            cv2.circle(visual_frame, (u, v), 6, (255, 0, 0), -1)
            cv2.putText(visual_frame, "Cube Center", (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        for cid, pos in gripper_positions.items():
            if pos is not None and pos[2] != 0:
                u = int(pos[0] * self.fx / pos[2] + self.cx)
                v = int(pos[1] * self.fy / pos[2] + self.cy)
                cv2.circle(visual_frame, (u, v), 6, (0, 0, 255), -1)
                cv2.putText(visual_frame, f"Gripper {cid}", (u + 8, v - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # --- Show in Unique Window per Camera ---
        visual_frame = cv2.resize(visual_frame, (960, 540))
        cv2.imshow(self.window_name, visual_frame)
        cv2.waitKey(1)


    def get_cube_center_relative(self):
        """Returns cube position in marker's coordinate frame (X, Y, Z), or None."""
        if self.last_origin is None or self.last_axes is None or self.last_cube_pos is None:
            return None
        R = np.stack(self.last_axes, axis=1)
        relative = self.last_cube_pos - self.last_origin
        return R.T @ relative

    def get_gripper_relative(self, gripper_id):
        """Returns gripper position in marker's coordinate frame (X, Y, Z), or (0,0,0) if not found."""
        if (self.last_origin is None or self.last_axes is None or
            gripper_id not in self.agent_gripper_positions):
            return np.zeros(3)

        pos = self.agent_gripper_positions[gripper_id]
        if pos is None or pos[2] == 0:  # not detected
            return np.zeros(3)

        R = np.stack(self.last_axes, axis=1)
        relative = pos - self.last_origin
        return R.T @ relative

    def stop(self):
        self.pipeline.stop()
