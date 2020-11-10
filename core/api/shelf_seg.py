from abc import abstractclassmethod, ABCMeta
from typing import Dict, Tuple, Sequence, List
import json

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

    def process(self, img) -> ShelfInfo:
        '''
        Process the image and get the segmentation result for the every row
        :param img:  numpy array image
        :return: the ShelfInfo
        '''
        pass