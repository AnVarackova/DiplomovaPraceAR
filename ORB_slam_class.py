# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 00:11:52 2021
The code was taken and modified from the following project on GitHub
https://github.com/yitao-yu/PythonORBSlAM
"""
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from copy import copy


class Frame(object):
    idx = 0
    last_kps, last_des, last_pose = None, None, None

    def __init__(self, image):
        Frame.idx += 1

        self.image = image
        self.idx = Frame.idx
        self.last_kps = Frame.last_kps
        self.last_des = Frame.last_des
        self.last_pose = Frame.last_pose

        self.curr_kps = None
        self.curr_des = None
        self.curr_pose = None
        self.points4d = None
        self.trans = None
        self.t = None
        self.h = None
        # self.per = None

    # Normalizing using internal parameters
    def normalize(K, pts):
        Kinv = np.linalg.inv(K)
        add_ones = lambda x: np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
        norm_pts = np.dot(Kinv, add_ones(pts).T).T[:,
                   0:2]
        return norm_pts

    def execute(self, params: dict):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if params["lowlight"]:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        orb = cv2.ORB_create(nfeatures=2000, nlevels=8, edgeThreshold=31)

        def cvCorners():
            pts = cv2.goodFeaturesToTrack(gray, params["maxfeatures"], qualityLevel=0.01, minDistance=3)
            features = [cv2.KeyPoint(x=pt[0][0], y=pt[0][1], _size=20) for pt in pts]
            return features

        def simpleORB():
            features = orb.detect(gray, mask=None)
            return features

        def evenORB():
            wh = np.shape(gray)[0:2]
            features = np.array([])
            orb = cv2.ORB_create(nfeatures=int(params["maxfeatures"]/pow(params["n"], 2)), nlevels=8, edgeThreshold=31)
            param = params["n"]
            for wsec in range(1, param):
                for hsec in range(1, param):
                    mask = np.zeros(np.shape(gray)[0:2], dtype=np.uint8)
                    mask[int(wh[0] * (wsec - 1) / param):int(wh[0] * wsec / param),
                    int(wh[1] * (hsec - 1) / param):int(wh[1] * hsec / param)] = 255
                    mask_features = orb.detect(gray, mask=mask)
                    features = np.append(features, mask_features)
            return features

        def evenCorners():
            wh = np.shape(gray)[0:2]
            param = params["n"]
            features = np.array([])
            for wsec in range(1, param + 1):
                for hsec in range(1, param + 1):
                    mask = np.zeros(np.shape(gray)[0:2], dtype=np.uint8)
                    mask[int(wh[0] * (wsec - 1) / param):int(wh[0] * wsec / param),
                    int(wh[1] * (hsec - 1) / param):int(wh[1] * hsec / param)] = 255
                    pts = cv2.goodFeaturesToTrack(gray, int(params["maxfeatures"] / pow(params["n"], 2)),
                                                  qualityLevel=0.01, minDistance=3, mask=mask)
                    if pts is None:
                        continue
                    mask_features = [cv2.KeyPoint(pt[0][0], pt[0][1], 20) for pt in pts]
                    features = np.append(features, mask_features)
            return features

        F = {"cvCorners": cvCorners,
             "simpleORB": simpleORB,
             "evenORB": evenORB,
             "evenCorners": evenCorners}[params["featureextract"]]
        features = F()

        kps, des = orb.compute(gray, features)

        Frame.last_kps = copy(kps)
        Frame.last_des = copy(des)

        if self.idx == 1:
            self.curr_pose = np.eye(4)
            self.points4d = None
            Frame.last_pose = self.curr_pose
            return self

        bfmatch = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bfmatch.knnMatch(des, self.last_des, k=2)
        matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        assert len(matches) >= 8

        kps = [kps[m.queryIdx] for m in matches]
        lastkps = [self.last_kps[m.trainIdx] for m in matches]

        kps_corr = np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in kps])
        lastkps_corr = np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in lastkps])
        K = params["K"]

        norm_curr_kps = Frame.normalize(K, kps_corr)
        norm_last_kps = Frame.normalize(K, lastkps_corr)

        model, inliers = ransac((norm_last_kps, norm_curr_kps),
                                EssentialMatrixTransform,
                                min_samples=8,
                                residual_threshold=0.005,
                                max_trials=200)

        E = model.params
        inliers_index = [i for i in range(0, len(inliers)) if inliers[i] == True]
        self.curr_kps = [kps[i] for i in inliers_index]
        self.last_kps = [lastkps[i] for i in inliers_index]
        kps_corr = kps_corr[inliers]
        lastkps_corr = lastkps_corr[inliers]
        ret, R, t, mask, pts = cv2.recoverPose(E, kps_corr, lastkps_corr, K, distanceThresh=params["maxdepth"])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.T
        trans = np.eye(4)
        trans[:3, :3] = R
        trans[:3, 3] = t.T * (-1)
        self.trans = trans
        self.t = t

        self.curr_pose = np.dot(T, self.last_pose)
        Frame.last_pose = self.curr_pose
        pts1 = Frame.normalize(K, lastkps_corr)
        pts2 = Frame.normalize(K, kps_corr)
        pose1 = np.linalg.inv(self.last_pose)
        pose2 = np.linalg.inv(self.curr_pose)

        points4d = cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
        points4d /= points4d[:, 3:]
        points4d = [p for p in points4d if p[2] > 0]
        self.points4d = points4d

        self.h, mask = cv2.findHomography(lastkps_corr, kps_corr)
        kps1 = np.float32(kps_corr[:, :3])
        kps2 = np.float32(lastkps_corr[:, :3])
        draw_points(self, kps_corr, lastkps_corr)

        return self


def draw_points(frame, kps_corr, lastkps_corr):
    for kp1, kp2 in zip(kps_corr, lastkps_corr):
        u1, v1 = int(kp1[0]), int(kp1[1])
        u2, v2 = int(kp2[0]), int(kp2[1])
        cv2.circle(frame.image, (u1, v1), color=(0, 0, 255), radius=3)
        cv2.line(frame.image, (u1, v1), (u2, v2), color=(255, 0, 0))
    return None
