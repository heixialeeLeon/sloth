from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
from core.api.offtake_detector import OfftakeInfo
from core.api.utils import check_bbox

class ShelfEngine(metaclass=ABCMeta):
    '''
    the main class for the shelf related function
    '''
    def __init__(self, shelf_adjust, offtake_det, shelf_match=None):
        self.shelf_adjust = shelf_adjust
        self.offtake_det = offtake_det
        self.shelf_match = shelf_match

    def diff_areas(self, img) -> OfftakeInfo:
        '''
        get the offtake infos
        :param img:
        :return:
        '''
        img_h, img_w = img.shape[:2]
        img_top = img[:img_h//2, :]
        img_bottom = img[img_h//2:, :]
        img_left = img[:, :img_w//2]
        adjust_img_top = self.shelf_adjust.process(img_top)
        adjust_img_bottom = self.shelf_adjust.process(img_bottom)
        top_result = self.offtake_det.process(adjust_img_top)
        bottom_result = self.offtake_det.process(adjust_img_bottom)

        if self.shelf_match is None:
            for item in bottom_result:
                item['bbox'][1] += img_h // 2
                item['bbox'][3] += img_h // 2
            return top_result + bottom_result
        else:
            match_result = self.shelf_match.process(img_left)
            bottom_result_2 = []
            for it_b in bottom_result:
                flag = False
                for i, b in enumerate(match_result["bottom"]):
                    if check_bbox(it_b["bbox"], b["polygon"]):
                        for it_t in top_result:
                            if check_bbox(it_t["bbox"], match_result["top"][i]["polygon"]):
                                flag = True
                                break
                    if flag:
                        break
                if not flag:
                    it_b['bbox'][1] += img_h // 2
                    it_b['bbox'][3] += img_h // 2
                    bottom_result_2.append(it_b)

            return top_result + bottom_result_2
