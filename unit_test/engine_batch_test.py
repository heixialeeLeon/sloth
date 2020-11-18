import sys
sys.path.append(".")
import cv2
import numpy as np
#from configs.config import shelf_model, similarity_model, offtake_model
from configs.unit_test_config import shelf_model, similarity_model, offtake_model
from core.api.shelf_seg import ShelfSegmentation
from core.api.shelf_adjust import ShelfAdjust
from core.api.image_similarity import ImageSimilarity
from core.api.offtake_detector import OfftakeDetector
from core.api.shelf_engine import ShelfEngine
from glob import glob

shelf_seg = ShelfSegmentation(shelf_model["model_config"], shelf_model["model_checkpoint"])
img_sim = ImageSimilarity(similarity_model["model_config"], similarity_model["model_checkpoint"])
shelf_adjust = ShelfAdjust(shelf_seg, img_sim)
offtake_det = OfftakeDetector(offtake_model["model_config"], offtake_model["model_checkpoint"])
engine = ShelfEngine(shelf_adjust, offtake_det)

test_folder = "/home/leon/data/offtake_val"

for img_path in glob(test_folder+"/*.jpg"):
    img = cv2.imread(str(img_path))
    result = engine.diff_areas(img)
    for item in result:
        bbox = item['bbox']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    cv2.imshow("diff", img)
    cv2.waitKey(0)
