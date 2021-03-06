# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
import numpy  as np


def circle_loss(
    sim_ap: torch.Tensor,
    sim_an: torch.Tensor,
    scale: float = 16.0,
    margin: float = 0.1,
    redection: str = "mean"
):
    pair_ap = -scale * (sim_ap - margin)
    pair_an = scale * sim_an
    pair_ap = torch.logsumexp(pair_ap, dim=1)
    pair_an = torch.logsumexp(pair_an, dim=1)
    loss = torch.nn.functional.softplus(pair_ap + pair_an)
    if redection == "mean":
        loss = loss.mean()
    elif redection == "sum":
        loss = loss.sum()
    return loss

@torch.no_grad()
def update_queue(queue, pointer, new_item):
    n = new_item.shape[0]
    length = queue.shape[0]
    if pointer + n <= length:
        queue[pointer: pointer + n] = new_item
        pointer = pointer + n
    else:
        res = n-(length-pointer)
        queue[pointer: length] = new_item[:length-pointer]
        queue[: res] = new_item[-res:]
        pointer = res
    return queue, pointer


class OIM(Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, queue, num_gt, momentum):
        ctx.lut = lut
        ctx.queue = queue
        ctx.num_gt = num_gt
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(ctx.lut.t())
        outputs_unlabeled = inputs.mm(ctx.queue.t())
        return torch.cat((outputs_labeled, outputs_unlabeled), 1)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_outputs, = grad_outputs
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((ctx.lut, ctx.queue), 0))

        for i, (x, y) in enumerate(zip(inputs, targets)):
            if y == -1:
                tmp = torch.cat((ctx.queue[1:], x.view(1, -1)), 0)
                ctx.queue[:, :] = tmp[:, :]
            elif 0 <= y < len(ctx.lut):
                if i < ctx.num_gt:
                    ctx.lut[y] = ctx.momentum * ctx.lut[y] + (1. - ctx.momentum) * x
                    ctx.lut[y] = F.normalize(ctx.lut[y], dim=-1)
            else:
                continue
        return grad_inputs, None, None, None, None, None


class OIMLossComputation(nn.Module):
    def __init__(self, cfg):
        super(OIMLossComputation, self).__init__()
        self.cfg = cfg.clone()

        if 'sysu' in self.cfg.DATASETS.TRAIN[0]:
            self.num_pid = 5532
            self.queue_size = 5000
        elif 'prw' in self.cfg.DATASETS.TRAIN[0]:
            self.num_pid = 483
            self.queue_size = 500
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.lut_momentum = 0.0
        self.out_channels = self.cfg.REID.OUT_CHANNELS

        self.register_buffer('lut', torch.zeros(self.num_pid, self.out_channels).cuda())
        self.register_buffer('queue', torch.zeros(self.queue_size, self.out_channels).cuda())

    def forward(self, features, results, targets):

        gt_box_pre_img = [len(i) for i in targets]
        features = [result.get_field('reid_feat') for result in results]
        pids = [result.get_field('pid') for result in results]

        gt_features = [feat[:n] for (n, feat) in zip(gt_box_pre_img, features)]
        de_features = [feat[n:] for (n, feat) in zip(gt_box_pre_img, features)]

        gt_pids = [pid[:n] for (n, pid) in zip(gt_box_pre_img, pids)]
        de_pids = [pid[n:] for (n, pid) in zip(gt_box_pre_img, pids)]

        features = torch.cat(gt_features+de_features)
        pids = torch.cat(gt_pids+de_pids)
        aux_label = pids  # threshold<0.7 pid=-2

        aux_label_np = aux_label.data.cpu().numpy()
        invalid_inds = np.where((aux_label_np < 0))
        aux_label_np[invalid_inds] = -1
        pid_label = torch.from_numpy(aux_label_np).long().cuda().view(-1)  # threshold<0.7 pid=-1

        num_gt = sum([len(i) for i in targets])

        reid_result = OIM.apply(features, aux_label, self.lut, self.queue, num_gt, self.lut_momentum)
        loss_weight = torch.cat([torch.ones(self.num_pid).cuda(), torch.zeros(self.queue_size).cuda()])

        scalar = 10
        loss_reid = F.cross_entropy(reid_result * scalar, pid_label, weight=loss_weight, ignore_index=-1)
        return loss_reid * self.cfg.REID.LOSS_SCALE


class CIRCLELossComputation(nn.Module):
    def __init__(self, cfg):
        super(CIRCLELossComputation, self).__init__()
        self.cfg = cfg.clone()

        if 'sysu' in self.cfg.DATASETS.TRAIN[0]:
            num_labeled = 8192
            num_unlabeled = 8192
        elif 'prw' in self.cfg.DATASETS.TRAIN[0]:
            num_labeled = 8192
            num_unlabeled = 8192
        else:
            raise KeyError(cfg.DATASETS.TRAIN)

        self.out_channels = self.cfg.REID.OUT_CHANNELS

        self.register_buffer('pointer', torch.zeros(2, dtype=torch.int).cuda())
        self.register_buffer('id_inx', -torch.ones(num_labeled, dtype=torch.long).cuda())
        self.register_buffer('lut', torch.zeros(num_labeled, self.out_channels).cuda())
        self.register_buffer('queue', torch.zeros(num_unlabeled, self.out_channels).cuda())

    def forward(self, features, results, targets):

        gt_box_pre_img = [len(i) for i in targets]
        features = [result.get_field('reid_feat') for result in results]
        pids = [result.get_field('pid') for result in results]

        gt_features = [feat[:n] for (n, feat) in zip(gt_box_pre_img, features)]
        de_features = [feat[n:] for (n, feat) in zip(gt_box_pre_img, features)]

        gt_pids = [pid[:n] for (n, pid) in zip(gt_box_pre_img, pids)]
        de_pids = [pid[n:] for (n, pid) in zip(gt_box_pre_img, pids)]

        # features = torch.cat(gt_features+de_features)
        # pids = torch.cat(gt_pids+de_pids)
        features = torch.cat(gt_features)
        pids = torch.cat(gt_pids)
        aux_label = pids  # threshold<0.7 pid=-2

        aux_label_np = aux_label.data.cpu().numpy()
        invalid_inds = np.where((aux_label_np < 0))
        aux_label_np[invalid_inds] = -1

        id_labeled = aux_label[aux_label > -1].to(torch.long)
        feat_labeled = features[aux_label > -1]
        feat_unlabeled = features[aux_label == -1]
        self.lut, _ = update_queue(self.lut, self.pointer[0], feat_labeled)

        self.id_inx, self.pointer[0] = update_queue(self.id_inx, self.pointer[0], id_labeled)
        self.queue, self.pointer[1] = update_queue(self.queue, self.pointer[1], feat_unlabeled)

        queue_sim = torch.mm(feat_labeled, self.queue.t())
        lut_sim = torch.mm(feat_labeled, self.lut.t())
        positive_mask = id_labeled.view(-1, 1) == self.id_inx.view(1, -1)
        sim_ap = lut_sim.masked_fill(~positive_mask, float("inf"))
        sim_an = lut_sim.masked_fill(positive_mask, float("-inf"))
        sim_an = torch.cat((queue_sim, sim_an), dim=-1)

        pair_loss = circle_loss(sim_ap, sim_an)
        return pair_loss


def make_reid_loss_evaluator(cfg):
    loss_evaluator = OIMLossComputation(cfg)
    # loss_evaluator = CIRCLELossComputation(cfg)
    return loss_evaluator