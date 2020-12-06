import sys
sys.path.append(".")
import cv2
import numpy as np
#from configs.config import shelf_model, similarity_model, offtake_model
from configs.unit_test_config import shelf_model, similarity_model, similarity2_model, offtake_model
from core.api.shelf_seg import ShelfSegmentation
from core.api.shelf_adjust import ShelfAdjust
from core.api.shelf_match import ShelfMatch
from core.api.image_similarity import ImageSimilarity
from core.api.offtake_detector import OfftakeDetector
from core.api.shelf_engine import ShelfEngine

shelf_seg = ShelfSegmentation(shelf_model["model_config"], shelf_model["model_checkpoint"])
img_sim = ImageSimilarity(similarity_model["model_config"], similarity_model["model_checkpoint"])
img_sim2 = ImageSimilarity(similarity2_model["model_config"], similarity2_model["model_checkpoint"])
shelf_adjust = ShelfAdjust(shelf_seg, img_sim)
shelf_match = ShelfMatch(shelf_seg, img_sim2)
offtake_det = OfftakeDetector(offtake_model["model_config"], offtake_model["model_checkpoint"])
engine = ShelfEngine(shelf_adjust, offtake_det, shelf_match)

img = cv2.imread("./test_imgs/engine_test.jpg")
result = engine.diff_areas(img)
for item in result:
    bbox = item['bbox']
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)

print(result)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey()
