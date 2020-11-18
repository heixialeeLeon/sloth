from easydict import EasyDict

prefix = "../"

shelf_model = EasyDict(dict(
    device = "cuda",
    model_config= prefix+"configs/shelf_mask_rcnn_r50_fpn.py",
    model_checkpoint= prefix+"models/shelf_maskrcnn_r50-9fb44b46.pth",
))

similarity_model =  EasyDict(dict(
    device = "cuda",
    model_config="mnas",
    model_checkpoint= prefix+"models/mnas.pth",
))

offtake_model = EasyDict(dict(
    device = "cuda",
    model_config= prefix+"configs/offtake_faster_rcnn_r50_fpn.py",
    model_checkpoint= prefix + "models/offtake_frcnn_r50-6b9c2a72.pth",
))