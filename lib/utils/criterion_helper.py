import torch.nn as nn


class FeatureMSELoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feature_rec = input["feature_rec"]
        feature_align = input["feature_align"]
        return self.criterion_mse(feature_rec, feature_align)


class ImageMSELoss(nn.Module):
    """Train separate decoders for visualization of reconstructed features"""

    def __init__(self, weight):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        image = input["image"]
        image_rec = input["image_rec"]
        return self.criterion_mse(image, image_rec)

class DiffusionLoss(nn.Module):
    ''' Directly use p_losses in GD
    '''
    def __init__(self, weight, loss_fn='MSELoss'):
        super().__init__()
        self.test_criterion = getattr(nn, loss_fn)()
        self.weight = weight

    def forward(self, input):
        if input.get('loss',None) is None:
            feature_rec = input["feature_rec"]
            feature_align = input["feature_align"]
            loss = self.test_criterion(feature_rec,feature_align)
        else:
            loss = input["loss"]

        return loss

def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
