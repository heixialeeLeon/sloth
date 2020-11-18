import sys
sys.path.append(".")
import cv2
import numpy as np
#from configs.config import shelf_model, similarity_model
from configs.unit_test_config import shelf_model, similarity_model
from core.api.shelf_seg import ShelfSegmentation
from core.api.shelf_adjust import ShelfAdjust
from core.api.image_similarity import ImageSimilarity


shelf_seg = ShelfSegmentation(shelf_model["model_config"], shelf_model["model_checkpoint"])
img_sim = ImageSimilarity(similarity_model["model_config"], similarity_model["model_checkpoint"])
shelf_adjust = ShelfAdjust(shelf_seg, img_sim)

img = cv2.imread("../test_imgs/adjust_test.jpg")
img_corr = shelf_adjust.process(img)

cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img", np.concatenate([img, img_corr], 0))
cv2.waitKey()
