from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
import numpy as np
import cv2

class ShelfAdjust(metaclass=ABCMeta):
    def __init__(self, shelf_seg, img_sim):
        self.shelf_seg = shelf_seg
        self.img_sim = img_sim
        self.shelf_size = (128, 32)
        self.pic_size = (256, 320)

    def process(self, img):
        '''
        Process the img and align the left and the right
        :param img: the img with left and right
        :return:  the aligned img with left and right
        '''
        img_h, img_w = img.shape[:2]
        img_l = img[:, :img_w // 2]
        img_r = img[:, img_w // 2:]
        left = self.shelf_seg.process(img_l)
        right = self.shelf_seg.process(img_r)
        left = [res for res in left if len(res["polygon"]) == 4]
        right = [res for res in right if len(res["polygon"]) == 4]
        Nl, Nr = len(left), len(right)
        if min(Nl, Nr) <= 1:
            return img

        cut_imgs = []
        cut_w, cut_h = self.shelf_size
        dst_pts = np.array([[0, 0], [cut_w - 1, 0], [cut_w - 1, cut_h - 1], [0, cut_h - 1]], dtype=np.float32)
        total_points = []
        for idx, res in enumerate(left + right):
            poly = res["polygon"]
            poly.sort(key=lambda x: x[0])
            tl, bl = sorted(poly[:2], key=lambda x: x[1])
            tr, br = sorted(poly[2:], key=lambda x: x[1])
            src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            total_points.append(src_pts)
            m = cv2.getPerspectiveTransform(src_pts, dst_pts)
            cut_img = cv2.warpPerspective(img_l if idx < Nl else img_r, m, self.shelf_size)
            cut_imgs.append(cut_img)

        features = self.img_sim.get_features(cut_imgs)
        feat_left = features[:Nl]
        feat_right = features[Nl:]
        sim_mat = self.img_sim.calc_similarity_matrix(feat_left, feat_right)
        mean_sim = sim_mat.mean()

        left_indices = []
        right_indices = []
        N = min(Nl, Nr)
        for _ in range(N):
            min_idx = np.argmin(sim_mat)
            i, j = divmod(min_idx, Nr)
            if i in left_indices or j in right_indices:
                break
            if sim_mat[i, j] > mean_sim:
                break
            left_indices.append(i)
            right_indices.append(j)
            sim_mat[i, :] = np.inf
            sim_mat[:, j] = np.inf

        left_pts = np.array([total_points[idx] for idx in left_indices]).reshape(-1, 2)
        right_pts = np.array([total_points[Nl + idx] for idx in right_indices]).reshape(-1, 2)
        H, _ = cv2.findHomography(right_pts, left_pts, cv2.LMEDS)
        img_rx = cv2.warpPerspective(img_r, H, (img_w // 2, img_h))

        cut_imgs = [cv2.resize(im, self.pic_size) for im in [img_l, img_r, img_rx]]
        features = self.img_sim.get_features(cut_imgs)
        feat_left = features[:1]
        feat_right = features[1:]
        sim_mat = self.img_sim.calc_similarity_matrix(feat_left, feat_right)[0]
        if sim_mat[0] / sim_mat[1] > 1.2:
            img_corr = np.concatenate([img_l, img_rx], 1)
            return img_corr
        else:
            return img
