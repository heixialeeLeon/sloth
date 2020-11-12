from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
import json
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from core.api.utils import remove_overlap, filter_results, PolygonFit

class RowInfo(object):
    def __init__(self, row_index: int, location: List):
        self.row_index = row_index
        self.location = location

class ShelfInfo(object):
    def __init__(self):
        self.shelf = list()

    def add_row(self, row: RowInfo) -> None:
        '''
        Add the row location
        :param row: the type of RowInfo
        :return:  None
        '''
        self.shelf.append(row)

    def to_json(self):
        return json.dumps(self, default=lambda o:o.__dict__, sort_keys=False)

class ShelfSegmentation(metaclass=ABCMeta):
    def __init__(self, config, checkpoint, score_thr=0.5):
        self.config = config
        self.checkpoint = checkpoint
        self.score_thr = 0.5
        self.model = init_detector(config, checkpoint)

    def process(self, img) -> ShelfInfo:
        '''
        Process the image and get the segmentation result for the every row
        :param img:  numpy array image
        :return: the ShelfInfo
        '''
        prediction = inference_detector(self.model, img)
        polygon_fit = PolygonFit()
        results = []
        for bbox, seg in zip(prediction[0][0], prediction[1][0]):
            if bbox[-1] < self.score_thr:
                continue
            contours, hierarchy = cv2.findContours(np.array(seg, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cv2.convexHull(c) for c in contours]
            contours = [cv2.approxPolyDP(c, 0.001 * cv2.arcLength(c, True), True) for c in contours]
            contours = [c for c in contours if c.shape[0] > 2]

            if len(contours) == 0:
                continue
            elif len(contours) > 1:
                contour_areas = [cv2.contourArea(c) for c in contours]
                max_idx = np.argmax(contour_areas)
                contour = contours[max_idx]
            else:
                contour = contours[0]
            contour = contour.reshape(-1, 2)
            polygon = polygon_fit.fit(contour)
            results.append(dict(
                bbox=bbox[:-1].round().astype(np.int32).tolist(),
                points=contour.tolist(),
                polygon=polygon,
                score=float(bbox[-1])
            ))
        results.sort(key=lambda x: x['bbox'][1])
        results = remove_overlap(results)
        results = filter_results(results)
        return results
