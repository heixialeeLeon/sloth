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
from utils.json_utils import read_json_file,get_bbox_from_json
from utils.metrics import get_metrics
from glob import glob

shelf_seg = ShelfSegmentation(shelf_model["model_config"], shelf_model["model_checkpoint"])
img_sim = ImageSimilarity(similarity_model["model_config"], similarity_model["model_checkpoint"])
shelf_adjust = ShelfAdjust(shelf_seg, img_sim)
offtake_det = OfftakeDetector(offtake_model["model_config"], offtake_model["model_checkpoint"], score_thr=0.6)
engine = ShelfEngine(shelf_adjust, offtake_det)

test_folder = "test_imgs/offtake_val"

def show_ground_gt(img, json_path):
    json_data = read_json_file(json_path)
    bboxes = get_bbox_from_json(json_data)
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)

def show_detect(img):
    result = engine.diff_areas(img)
    for item in result:
        bbox = item['bbox']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)

def show_detect_gt_result():
    for img_path in glob(test_folder+"/*.jpg"):
        json_path = str(img_path)[:-3] + "json"
        img = cv2.imread(str(img_path))
        show_ground_gt(img, json_path)
        show_detect(img)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        cv2.imshow("diff", img)
        cv2.waitKey(0)

def show_detect_gt_result_split():
    for img_path in glob(test_folder+"/*.jpg"):
        json_path = str(img_path)[:-3] + "json"
        img_det = cv2.imread(str(img_path))
        img_gt = img_det.copy()
        show_ground_gt(img_gt, json_path)
        show_detect(img_det)
        img = np.vstack((img_gt,img_det))
        img = cv2.resize(img, None, fx=0.3, fy=0.3)
        cv2.imshow("diff", img)
        cv2.waitKey(0)

def test_detect_metrics():
    tps, fns, fps = 0, 0, 0
    for img_path in glob(test_folder+"/*.jpg"):
        json_path = str(img_path)[:-3] + "json"
        img = cv2.imread(str(img_path))
        results = engine.diff_areas(img)
        predictions = [item['bbox'] for item in results]
        ground_truths = get_bbox_from_json(read_json_file(json_path))
        tp, fn, fp = get_metrics(predictions, ground_truths, iou_thres=0.3)
        tps += tp
        fns += fn
        fps += fp
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"precision: {precision:.4f}, recall: {recall:.4f}, f1_score: {f1_score:.4f}")

if __name__ == "__main__":
    # show_detect_gt_result()
    # show_detect_gt_result_split()
    test_detect_metrics()

