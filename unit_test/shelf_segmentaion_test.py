import sys
sys.path.append(".")
import cv2
import numpy as np
#from configs.config import shelf_model
from configs.unit_test_config import shelf_model
from core.api.shelf_seg import ShelfSegmentation


shelf_seg = ShelfSegmentation(shelf_model["model_config"], shelf_model["model_checkpoint"])
img = cv2.imread("../test_imgs/shelf_test.jpg")
result = shelf_seg.process(img)
for item in result:
    poly = np.array(item['polygon']).reshape(-1, 1, 2)
    cv2.drawContours(img, [poly], -1, (0, 0, 255), 3)

print(result)
cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.imshow("img", img)
cv2.waitKey()
