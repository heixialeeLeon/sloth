from easydict import EasyDict

shelf_model = EasyDict(dict(
    device = "cuda",
    model_config="configs/shelf_mask_rcnn_r50_fpn.py",
    model_checkpoint= "models/shelf_maskrcnn_r50-9fb44b46.pth",
))

similarity_model =  EasyDict(dict(
    device = "cuda",
    model_config="mnas",
    model_checkpoint= "models/mnas.pth",
))

offtake_model = EasyDict(dict(
    device = "cuda",
    model_config="configs/offtake_faster_rcnn_r50_fpn.py",
    model_checkpoint= "models/offtake_frcnn_r50-6b9c2a72.pth",
))