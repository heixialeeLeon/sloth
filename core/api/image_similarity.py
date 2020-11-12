import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


class ImageSimilarity(object):
    def __init__(self, config, checkpoint):
        if config == "res18":
            self.model = nn.Sequential(*list(models.resnet18().children())[:-2])
        elif config == "res50":
            self.model = nn.Sequential(*list(models.resnet50().children())[:-2])
        elif config == "mbv2":
            self.model = models.mobilenet_v2().features
        elif config == "mnas":
            self.model = models.mnasnet1_0().layers
        else:
            print("Unsupported model {}".format(model))
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        self.model = self.model.cuda()
        self.model.eval()
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_features(self, images):
        data = torch.stack([self.transforms(img) for img in images], 0).cuda()
        with torch.no_grad():
            return self.model(data).mean(3).mean(2)

    def calc_similarity_matrix(self, src, dst):
        with torch.no_grad():
            src = src.flatten(1).unsqueeze(1)
            dst = dst.flatten(1).unsqueeze(0)

            diff = ((src - dst) ** 2).sum(2).sqrt()
            return diff.detach().cpu().numpy()
