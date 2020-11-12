from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
import json
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector

class OfftakeArea(object):
    def __init__(self, type: int, location: List):
        self.type = type
        self.location = location

class OfftakeInfo(object):
    def __init__(self):
        self.diff = list()

    def add_area(self, area: OfftakeArea):
        '''
        add the diff area record
        :param area:
        :return:
        '''
        self.diff.append(area)

    def to_json(self):
        return json.dumps(self, default=lambda o:o.__dict__, sort_keys=False)

class OfftakeDetector(metaclass=ABCMeta):
    def __init__(self, config, checkpoint, score_thr=0.5):
        self.config = config
        self.checkpoint = checkpoint
        self.score_thr = score_thr
        self.model = init_detector(config, checkpoint)

    def process(self, img) -> OfftakeInfo:
        '''
        detect the different areas for the left and right img
        :param img: the img with left and right
        :return: the different areas 
        '''
        prediction = inference_detector(self.model, img)[0]
        results = []
        for res in prediction:
            bbox = res[:-1].round().astype(np.int32).tolist()
            score = float(res[-1])
            if score > self.score_thr:
                results.append(dict(
                    bbox=bbox,
                    score=score
                ))
        results.sort(key=lambda x: x['bbox'][1])
        return results
