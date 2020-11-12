import sys
sys.path.append(".")
import cv2
import numpy as np
from configs.config import offtake_model
from core.api.offtake_detector import OfftakeDetector


offtake_det = OfftakeDetector(offtake_model["model_config"], offtake_model["model_checkpoint"])
img = cv2.imread("test_imgs/offtake_test.jpg")
result = offtake_det.process(img)
for item in result:
    bbox = item['bbox']
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)

print(result)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey()
