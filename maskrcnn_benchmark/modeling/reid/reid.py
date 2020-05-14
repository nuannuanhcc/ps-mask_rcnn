# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from .reid_feature_extractors import make_reid_feature_extractor
from .loss import make_reid_loss_evaluator

class REIDModule(torch.nn.Module):
    """
    Module for RPN computation. Takes feature maps from the backbone and outputs
    RPN proposals and losses. Works for both FPN and non-FPN.
    """

    def __init__(self, cfg):
        super(REIDModule, self).__init__()

        self.cfg = cfg.clone()

        self.feature_extractor = make_reid_feature_extractor(cfg)
        self.loss_evaluator = make_reid_loss_evaluator(cfg)

    def forward(self, features, results, targets):
        if not self.cfg.MODEL.RETINANET_ON and not self.training and targets is None:
            inds = [result.get_field('index') for result in results]
            feats = torch.cat([feat[ind] for (feat, ind) in zip(features, inds)])
        else:
            feats = features

        feats = self.feature_extractor(feats, results)
        feats = F.normalize(feats, dim=-1)

        num_feats = 0
        for result in results:
            l_res = len(result)
            result.add_field("reid_feat", feats[num_feats: num_feats+l_res])
            num_feats += l_res

        if not self.training:
            return feats, results, {}

        loss_reid = self.loss_evaluator(feats, results, targets)
        losses = {"loss_reid": loss_reid, }
        return feats, results, losses

def build_reid(cfg):
    """
    This gives the gist of it. Not super important because it doesn't change as much
    """
    return REIDModule(cfg)