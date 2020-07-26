# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

class RETINANETBOXComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """
    def __init__(
        self,
        cfg,
        proposal_matcher,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.retina_subsample_top_n = cfg.MODEL.RETINANET.SUBSAMPLE_TOP_N

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "pid"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        pids = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            pids_per_image = matched_targets.get_field("pid")
            pids_per_image = pids_per_image.to(dtype=torch.int64)

            pids_per_image[bg_inds] = -2
            pids_per_image[ignore_inds] = -2
            pids.append(pids_per_image)

            labels.append(labels_per_image)

        return labels, pids

    def __call__(self, proposals, targets):
        if [len(i) for i in proposals] == [len(i) for i in targets]:
            return proposals
        if self.retina_subsample_top_n > 0:
            gt_sizes = [len(box) for box in targets]
            box_sizes = [len(box) for box in proposals]
            if sum(gt_sizes) >= self.retina_subsample_top_n:
                proposals = [box[:n] for box, n in zip(proposals, gt_sizes)]
                return proposals
            elif sum(box_sizes) > self.retina_subsample_top_n:
                scores = torch.cat([box.get_field("scores") for box in proposals], dim=0)
                _, inds_sorted = torch.topk(scores, self.retina_subsample_top_n, dim=0, sorted=True)
                bool_type = torch.bool if float(torch.__version__[:3]) >= 1.2 else torch.uint8
                inds_mask = torch.zeros_like(scores, dtype=bool_type)
                inds_mask[inds_sorted] = 1
                inds_mask = inds_mask.split(box_sizes)
                for i in range(len(proposals)):
                    proposals[i] = proposals[i][inds_mask[i]]

        labels, pids = self.prepare_targets(proposals, targets)
        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, proposals_per_image, pids_per_image in zip(
            labels, proposals, pids
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("pid", pids_per_image)
        return proposals


def make_retinanet_box_subsample(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )
    return RETINANETBOXComputation(cfg, matcher,)
