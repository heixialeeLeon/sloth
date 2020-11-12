import math
import cv2
import numpy as np
from shapely.geometry import Polygon


def inter_over_min(A, B):
    polyA = Polygon(A).convex_hull
    polyB = Polygon(B).convex_hull
    if not polyA.intersects(polyB):
        return 0
    return float(polyA.intersection(polyB).area) / min(polyA.area, polyB.area)

def inter_over_union(A, B):
    try:
        polyA = Polygon(A).convex_hull
        polyB = Polygon(B).convex_hull
        if not polyA.intersects(polyB):
            return 0
        inter_area = polyA.intersection(polyB).area
        return float(inter_area) / (polyA.area + polyB.area - inter_area)
    except:
        return 0

def remove_overlap(results):
    N = len(results)
    overlap = []
    for i in range(N - 1):
        for j in range(i + 1, N):
            ov = inter_over_min(results[i]["polygon"], results[j]["polygon"])
            if ov > 0.5:
                overlap.append((i, j, ov))
    if len(overlap) > 0:
        need_remove = set()
        for i, j, ov in overlap:
            if results[i]["score"] > results[j]["score"]:
                need_remove.add(j)
            else:
                need_remove.add(i)
        results = [results[i] for i in range(N) if i not in need_remove]
    
    return results

def overlap_1d(A, B):
    x1, y1 = A
    x2, y2 = B
    x = max(x1, x2)
    y = min(y1, y2)
    if x >= y:
        return 0
    else:
        return float(y - x) / min(y1 - x1, y2 - x2)

def filter_results(results):
    if len(results) <= 1:
        return results
    groups = []
    spans = []
    for res in results:
        polygon = np.array(res["polygon"])
        xmin, xmax = polygon[:, 0].min(), polygon[:, 0].max()
        if len(spans) == 0:
            spans.append((xmin, xmax))
            groups.append([res])
        else:
            flag = False
            for idx, span in enumerate(spans):
                if overlap_1d(span, (xmin, xmax)) > 0.5:
                    spans[idx] = (min(span[0], xmin), max(span[1], xmax))
                    groups[idx].append(res)
                    flag = True
                    break
            if not flag:
                spans.append((xmin, xmax))
                groups.append([res])
    if len(groups) == 1:
        return groups[0]
    group_len = [len(g) for g in groups]
    idx = np.argmax(group_len)
    return groups[idx]

class PolygonFit(object):
    def fitLine(self, points):
        n = len(points)
        x, y = zip(*points)
        x, y = np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)
        mX, mY = x.mean(), y.mean()
        sXX = np.sum((x - mX) ** 2)
        sYY = np.sum((y - mY) ** 2)
        sXY = np.sum((x - mX) * (y - mY))
        
        isVertical = sXY == 0 and sXX < sYY
        isHorizontal = sXY == 0 and sXX > sYY
        isIndeterminate= sXY == 0 and sXX == sYY
        
        if isVertical:
            a, b, c = 1.0, 0.0, -mX
        elif isHorizontal:
            a, b, c = 0.0, 1.0, -mY
        elif isIndeterminate:
            a, b, c = np.NaN, np.NaN, np.NaN
        else:
            slope = (sYY - sXX + math.sqrt((sYY - sXX) * (sYY - sXX) + 4.0 * sXY * sXY)) / (2.0 * sXY)
            intercept = mY - slope * mX
            kFactors = 1 if intercept >= 0 else -1
            normFactor = kFactors * math.sqrt(slope * slope + 1.0)
            a = (float)(slope / normFactor)
            b = (float)(-1.0 / normFactor)
            c = (float)(intercept / normFactor)
        return (a, b, c)

    def linePointDistance(self, line, point):
        a, b, c = line
        x, y = point
        l = math.sqrt(a * a + b * b)
        dist = math.fabs(a * x + b * y + c) / l
        return dist
        
    def pointsDistance(self, pt1, pt2):
        x1, y1 = pt1
        x2, y2 = pt2
        dx, dy = x1 - x2, y1 - y2
        return math.sqrt(dx * dx + dy * dy)
    
    def lineIntersection(self, line1, line2):
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        m = a1 * b2 - a2 * b1
        if m == 0:
            return None
        x = (c2 * b1 - c1 * b2) / m
        y = (c1 * a2 - c2 * a1) / m
        x, y = int(round(x)), int(round(y))
        return (x, y)

    def fitEdge(self, points):
        arclen = cv2.arcLength(np.array(points, dtype=np.int32).reshape(-1, 1, 2), True)
        n = len(points)
        edges = [{'points': [points[i-1], points[i]],'length': self.pointsDistance(points[i-1], points[i])} for i in range(n)]

        while len(edges) > 2:
            err = [edges[i-1]['length'] + edges[i]['length'] - self.pointsDistance(edges[i-1]['points'][0], edges[i]['points'][-1]) for i in range(len(edges))]
            i = np.argmin(err)
            if err[i] > 0.01 * arclen:
                break
            if i == 0:
                edges = [{'points': edges[-1]['points'] + edges[0]['points'][1:],'length': edges[-1]['length'] + edges[0]['length']}] + edges[1:-1]
            elif i == len(edges) - 1:
                edges = edges[:i-1] + [{'points': edges[i-1]['points'] + edges[i]['points'][1:],'length': edges[i-1]['length'] + edges[i]['length']}]
            else:
                edges = edges[:i-1] + [{'points': edges[i-1]['points'] + edges[i]['points'][1:],'length': edges[i-1]['length'] + edges[i]['length']}] + edges[i+1:]

        lines = [self.fitLine(e['points']) for e in edges]
        cross = [self.lineIntersection(lines[i-1], lines[i]) for i in range(len(lines))]
        cross = [c for c in cross if c is not None]
        return cross
    
    def approxPoly(self, points):
        cnt = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        eps = 0.005
        arclen = cv2.arcLength(cnt, True)
        new_cnt = cv2.approxPolyDP(cnt, eps * arclen, True)
        while new_cnt.shape[0] > 4 and eps < 0.05:
            eps *= 1.2
            new_cnt = cv2.approxPolyDP(cnt, eps * arclen, True)
        points = new_cnt.reshape(-1, 2).tolist()
        return points

    def fit(self, points):
        approx = self.fitEdge(points)
        iou = inter_over_union(points, approx)
        if iou < 0.8:
            return self.approxPoly(points)
        return approx
