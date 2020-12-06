from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
import numpy as np
import cv2

class ShelfMatch(metaclass=ABCMeta):
    def __init__(self, shelf_seg, img_sim):
        self.shelf_seg = shelf_seg
        self.img_sim = img_sim
        self.shelf_size = (128, 32)
        self.dist_thres = 0.5

    def process(self, img):
        '''
        Process the img and align the left and the right
        :param img: the img with left and right
        :return:  the aligned img with left and right
        '''
        img_h, img_w = img.shape[:2]
        img_t = img[:img_h//2, :]
        img_b = img[img_h//2:, :]
        top = self.shelf_seg.process(img_t)
        bottom = self.shelf_seg.process(img_b)
        top = [res for res in top if len(res["polygon"]) == 4]
        bottom = [res for res in bottom if len(res["polygon"]) == 4]
        Nt, Nb = len(top), len(bottom)
        if min(Nt, Nb) <= 1:
            return None

        cut_imgs = []
        cut_w, cut_h = self.shelf_size
        dst_pts = np.array([[0, 0], [cut_w - 1, 0], [cut_w - 1, cut_h - 1], [0, cut_h - 1]], dtype=np.float32)
        total_points = []
        for idx, res in enumerate(top + bottom):
            poly = res["polygon"]
            poly.sort(key=lambda x: x[0])
            tl, bl = sorted(poly[:2], key=lambda x: x[1])
            tr, br = sorted(poly[2:], key=lambda x: x[1])
            src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
            total_points.append(src_pts)
            m = cv2.getPerspectiveTransform(src_pts, dst_pts)
            cut_img = cv2.warpPerspective(img_t if idx < Nt else img_b, m, self.shelf_size)
            cut_imgs.append(cut_img)

        features = self.img_sim.get_features(cut_imgs)
        feat_top = features[:Nt]
        feat_bottom = features[Nt:]
        sim_mat = self.img_sim.calc_similarity_matrix(feat_top, feat_bottom)
        min_idx = np.argmin(sim_mat)
        i, j = divmod(min_idx, Nb)
        if sim_mat[i, j] < self.dist_thres:
            result = {"top": [], "bottom": []}
            while i > 0 and j > 0:
                i -= 1
                j -= 1
            while i < Nt and j < Nb:
                result["top"].append(top[i])
                result["bottom"].append(bottom[j])
                i += 1
                j += 1
            return result
        return None
