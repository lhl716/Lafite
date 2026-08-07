import copy
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import numpy as np

from ..utils.misc_helper import to_device


class ModelHelper(nn.Module):
    """Build model from cfg"""

    def __init__(self, cfg):
        super(ModelHelper, self).__init__()

        self.frozen_layers = []
        for cfg_subnet in cfg:
            mname = cfg_subnet["name"]
            kwargs = cfg_subnet["kwargs"]
            mtype = cfg_subnet["type"]
            if cfg_subnet.get("frozen", False):
                self.frozen_layers.append(mname)
            if cfg_subnet.get("prev", None) is not None:
                prev_module = getattr(self, cfg_subnet["prev"])
                kwargs["inplanes"] = prev_module.get_outplanes()
                kwargs["instrides"] = prev_module.get_outstrides()

            module = self.build(mtype, kwargs)
            self.add_module(mname, module)

    def build(self, mtype, kwargs):
        module_name, cls_name = mtype.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**kwargs)

    def cuda(self):
        self.device = torch.device("cuda")
        return super(ModelHelper, self).cuda()

    def cpu(self):
        self.device = torch.device("cpu")
        return super(ModelHelper, self).cpu()

    def forward(self, input, membank=None, n_neighbos=1, neck_stop=False, **kwargs):
        if membank is None:
            input = copy.copy(input)
            if input["image"].device != self.device:
                input = to_device(input, device=self.device)
            for submodule in self.children():
                if neck_stop and submodule == self.reconstruction:
                    break
                output = submodule(input)
                input.update(output)
            return input
        else:
            return self.membank_inference(input, membank, n_neighbos, **kwargs)

    def membank_inference(self, inp, membank, n_neighbos=1, only_membank=False):
        inp = copy.copy(inp)
        if inp["image"].device != self.device:
            inp = to_device(inp, device=self.device)
        for submodule in self.children():
            if submodule == self.reconstruction:
                print('Pause before reconstruction.')
                break
            output = submodule(inp)
            inp.update(output)
        feature_align = inp['feature_align']
        origin_shape = feature_align.shape
        feature_numpy = rearrange(feature_align, 'N C H W -> (N H W) C').cpu().numpy()

        score, index = membank.search(feature_numpy, k=n_neighbos)
        index = np.asarray(index.astype(np.int64))
        if n_neighbos == 1:
            feature_masked = np.array([membank.reconstruct(int(itm[0])) for itm in index])
            feature_masked = torch.from_numpy(feature_masked) \
                    .to(feature_align.device, feature_align.dtype)
            feature_tensor = rearrange(feature_masked, '(N H W) C -> N C H W',
                    N=origin_shape[0], H=origin_shape[2], W=origin_shape[3])
        else:
            # cal weight for sum
            score_map = torch.from_numpy(score) \
                    .to(feature_align.device, feature_align.dtype)
            score_map = rearrange(score_map, '(N H W) K -> N K H W',
                    N=origin_shape[0], H=origin_shape[2], W=origin_shape[3])
            # K softmax
            weights = 1 - F.softmax(score_map, dim=1)
            ret = torch.zeros_like(feature_align)
            for k_idx in range(n_neighbos):
                feat_k_list = np.array([membank.reconstruct(int(itm[k_idx])) for itm in index])
                feat_k_list = torch.from_numpy(feat_k_list) \
                        .to(feature_align.device, feature_align.dtype)
                feat_k_tensor = rearrange(feat_k_list, '(N H W) C -> N C H W',
                        N=origin_shape[0], H=origin_shape[2], W=origin_shape[3])
                ret = ret + weights[:, k_idx].unsqueeze(1) * feat_k_tensor
            feature_tensor = ret

        def get_pred(inp_dict):
            pred = torch.sqrt(
                torch.mean((
                    inp_dict['feature_align'] - inp_dict['feature_rec']) ** 2,
                          dim=1, keepdim=True ))
            with torch.no_grad():
                pred = self.reconstruction.upsample(pred)
            inp_dict['pred'] = pred

        if only_membank:
            inp['feature_align'] = feature_align
            inp['feature_rec'] = feature_tensor
            get_pred(inp)
            return inp

        mask = torch.zeros_like(feature_tensor)
        inp['feature_align'] = feature_tensor

        output = self.reconstruction(inp, mask=mask)
        inp.update(output)

        inp['feature_align'] = feature_align
        get_pred(inp)

        return inp

    def ema_update(self):
        for submodule in self.children():
            if getattr(submodule, 'ema_update', None):
                submodule.ema_update()

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self
