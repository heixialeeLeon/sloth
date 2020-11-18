def inter_over_union(A, B):
    x1 = max(A[0], B[0])
    y1 = max(A[1], B[1])
    x2 = min(A[2], B[2])
    y2 = min(A[3], B[3])
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return 0
    s = w * h
    sa = (A[2] - A[0]) * (A[3] - A[1])
    sb = (B[2] - B[0]) * (B[3] - B[1])
    return float(s) / (sa + sb - s)

def get_metrics(predictions, ground_truths, iou_thres=0.3):
    tp, fn, fp = 0, 0, 0
    for pred in predictions:
        flag = False
        for gt in ground_truths:
            if inter_over_union(pred, gt) > iou_thres:
                flag = True
        if not flag:
            fp += 1
        else:
            tp += 1
            
    for gt in ground_truths:
        flag = False
        for pred in predictions:
            if inter_over_union(pred, gt) > iou_thres:
                flag = True
        if not flag:
            fn += 1
    return tp, fn, fp
