import json

def read_json_file(json_file):
    with open(json_file, "r") as f:
        data = f.read()
    obj = json.loads(data)
    return obj

def get_bbox_from_json(json_data):
    bboxes = list()
    for item in json_data["shapes"]:
        bbox = list()
        bbox.append(item["points"][0][0])
        bbox.append(item["points"][0][1])
        bbox.append(item["points"][1][0])
        bbox.append(item["points"][1][1])
        bboxes.append(bbox)
    return bboxes
