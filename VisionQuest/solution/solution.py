"""Eyes Over the Field: your solution.

Fill in the TODOs. Do not rename the class or the three methods.
The task is described in docs/PS.md; the exact data formats and conventions are in docs/SPEC.md.

How it is used:
    sol = Solution(camera_info, rules, info)     # once, before the flight
    sol.on_frame(frame, telemetry)               # for every camera tick (10 per second)
    answers = sol.finalize()                     # once, after the flight

Allowed packages: Python standard library (no file, OS or network access), numpy, opencv.
Helper files placed in the same folder can be imported.
"""
import numpy as np
import cv2


class Solution:
    def __init__(self, camera_info, rules, info):
        """Called once before the flight.

        camera_info: dict with width, height, K, dist, R_body_cam, fps    (docs/SPEC.md)
        rules:       dict with dictionary, valid_ids, marker_size_m, field_half_size_m
        info:        dict with part ("A", "B" or "C"), frame_rate, altitude_m, speed_mps
        """
        self.camera_info = camera_info
        self.rules = rules
        self.info = info

        # TODO: set up anything you need during the flight
        self.found = {}
        self.adjusted = {}
        self.sizes = {}
        self.pos = None
        self.last_t = None
        self.rej = 0
        self.drift = np.array([0, 0], dtype="float64")
        self.anchors = {}
        self.twins = set()
        self.valid_ids = set(int(i) for i in rules["valid_ids"])
        self.marker_size = float(rules.get("marker_size_m", 0.5))
        self.W = int(camera_info.get("width", 640))
        self.H = int(camera_info.get("height", 480))

        self.MIN_OBS = 5
        self.EDGE_MARGIN = 8
        self.CLUSTER_RADIUS = 1.5
        self.SAME_FRAME_SEP = 1.0
        self.SIZE_TOL = 0.20

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector_params.errorCorrectionRate = 0.5
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

    def rot_wb(self, roll, pitch, yaw):
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def pixels_to_world(self, pts_uv, tel):
        K = self.camera_info["K"]
        dist = self.camera_info["dist"]
        R_bc = self.camera_info["R_body_cam"]
        p = np.array([tel["x"], tel["y"], tel["z"]])
        R_wc = self.rot_wb(tel["roll"], tel["pitch"], tel["yaw"]) @ R_bc

        n = cv2.undistortPoints(np.asarray(pts_uv, np.float64).reshape(-1, 1, 2), np.array(K), np.array(dist))
        n = n.reshape(-1, 2)

        d = (R_wc @ np.column_stack([n, np.ones(len(n))]).T).T
        if np.any(d[:, 2] >= -1e-6):
            return None

        s = -p[2] / d[:, 2]

        return (s[:, None] * d)[:, :2]

    def smooth(self, tel):
        cur = np.array([tel["x"], tel["y"], tel["z"]], float)
        v = np.array([tel["vx"], tel["vy"], tel["vz"]], float)
        if self.pos is None:
            self.pos, self.last_t = cur, tel["t"]
            return
        dt = tel["t"] - self.last_t
        self.last_t = tel["t"]

        self.pos = self.pos + v * dt
        if np.linalg.norm(self.pos - cur) < 1:
            self.pos += 0.2 * (cur - self.pos)
            self.rej = 0
        else:
            self.rej += 1
            if self.rej >= 5:
                self.pos, self.rej = cur, 0

    def anchor(self, x, y, min_count=5, max_spread=0.5):
        pos = np.column_stack((x, y))
        if len(pos) < min_count:
            return None
        m = np.median(pos, axis=0)
        if np.median(np.linalg.norm(pos - m, axis=1)) > max_spread:
            return None
        return m

    def update(self, cur):
        d = []
        for i in cur:
            if i in self.anchors and i not in self.twins:
                d.append(self.anchors[i] - np.array(cur[i]))
        if d:
            offset = np.mean(np.array(d), axis=0)
            self.drift += (offset - self.drift) * 0.2
        for i in cur:
            self.adjusted[i][0].append(cur[i][0] + self.drift[0])
            self.adjusted[i][1].append(cur[i][1] + self.drift[1])

    def trusted(self, Id):
        if Id in self.twins:
            return False
        if abs(np.median(self.sizes[Id][-30:]) / self.marker_size - 1.0) > self.SIZE_TOL:
            return False

    def cluster(self, xy):
        labels = -np.ones(len(xy), int)
        centres = []
        for i, p in enumerate(xy):
            if centres:
                d = np.linalg.norm(np.array(centres) - p, axis=1)
                j = int(np.argmin(d))
                if d[j] < self.CLUSTER_RADIUS:
                    labels[i] = j
                    centres[j] = np.median(xy[labels == j], axis=0)
                    continue
            centres.append(p.copy())
            labels[i] = len(centres) - 1
        return labels
    
    def on_frame(self, frame, telemetry):
        """Called for every camera tick.

        frame:     BGR image, numpy uint8 array of shape (480, 640, 3),
                   or None when the camera dropped this frame
        telemetry: dict with t, x, y, z, roll, pitch, yaw, vx, vy, vz    (docs/SPEC.md)
        The return value is ignored.
        """
        self.smooth(telemetry)

        if frame is None:
            return

        telemetry = dict(telemetry, x=self.pos[0], y=self.pos[1], z=self.pos[2])

        # TODO: find the markers visible in this frame.
        corners, ids, rejected = self.detector.detectMarkers(frame)
        if ids is None:
            return

        # TODO: work out where each of them is on the ground (x, y in meters).
        cur = {}
        m = self.EDGE_MARGIN
        for i in range(len(ids)):
            Id = int(ids[i])
            if Id not in self.valid_ids:
                continue
            pt = corners[i].reshape(4, 2)
            if pt.min() < m or pt[:, 0].max() > self.W - m or pt[:, 1].max() > self.H - m:
                continue

            w = self.pixels_to_world(pt, telemetry)
            if w is None:
                continue

            w = w + self.pos[:2]
            a = w.mean(axis=0)
            side = np.linalg.norm(w - np.roll(w, -1, axis=0), axis=1).mean()

            if Id in cur and np.linalg.norm(np.array(cur[Id]) - a) > self.SAME_FRAME_SEP:
                self.twins.add(Id)

            # TODO: store whatever you will need in finalize().
            if Id not in self.found:
                self.found[Id] = [[], []]
                self.adjusted[Id] = [[], []]
                self.sizes[Id] = []
            self.found[Id][0].append(a[0])
            self.found[Id][1].append(a[1])
            self.sizes[Id].append(side)
            cur[Id] = (a[0], a[1])

            if self.trusted(Id):
                pos = self.anchor(self.found[Id][0], self.found[Id][1])
                if pos is not None:
                    self.anchors[Id] = pos
            else:
                self.anchors.pop(Id, None)

        self.update(cur)

    def finalize(self):
        """Called once after the flight.

        Return one entry per GENUINE marker: [(marker_id, x, y), ...] with x, y in meters.
        Fake markers and duplicate rows cost points (see docs/PS.md, Evaluation).
        """
        answers = []

        # TODO: decide which markers are genuine and where each one is, and fill `answers`.
        cand = {}
        for Id in self.adjusted:
            if Id in self.twins:
                continue
            xy = np.column_stack(self.adjusted[Id])
            labels = self.cluster(xy)
            strong = [l for l in np.unique(labels) if np.sum(labels == l) >= self.MIN_OBS]
            if len(strong) != 1:
                continue
            cand[Id] = labels == strong[0]

        med_size = {i: float(np.median(np.array(self.sizes[i])[cand[i]])) for i in cand}
        scale = 1.0
        if len(med_size) >= 5:
            scale = float(np.clip(np.median(list(med_size.values())) / self.marker_size, 0.8, 1.25))
        expected = self.marker_size * scale

        for Id in cand:
            if abs(med_size[Id] / expected - 1.0) > self.SIZE_TOL:
                continue
            mask = cand[Id]
            adj = np.column_stack(self.adjusted[Id])[mask]
            p = np.median(adj, axis=0)
            answers.append((Id, float(p[0]), float(p[1])))

        return answers