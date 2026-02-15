# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Loss functions."""

import math

import torch
import torch.nn as nn

from utils.torch_utils import de_parallel


def wing_loss(pred, target, w=0.5, epsilon=0.1):
    """Wing Loss for keypoint regression. More sensitive to small errors than L1/L2 loss.

    Args:
        pred: predicted keypoints (N, 8) - values typically in [-0.5, 1.5] range
        target: target keypoints (N, 8)
        w: wing width, controls the range of nonlinear part (adjusted for small coords)
        epsilon: curvature, smaller = more sensitive to small errors

    Returns:
        Wing loss value
    """
    diff = torch.abs(pred - target)
    c = w * (1.0 - math.log(1.0 + w / epsilon))

    # Wing loss: log for small errors, linear for large errors
    loss = torch.where(diff < w, w * torch.log(1.0 + diff / epsilon), diff - c)
    return loss.mean()


def polygon_giou_loss(pred_kpts, target_kpts):
    """Polygon GIoU Loss using bounding box approximation. Optimizes overall shape alignment, not just individual
    points.

    Args:
        pred_kpts: predicted keypoints (N, 8) - x1,y1,x2,y2,x3,y3,x4,y4
        target_kpts: target keypoints (N, 8)

    Returns:
        1 - GIoU (loss to minimize)
    """
    # Extract x and y coordinates
    pred_xs = pred_kpts[:, 0::2]  # (N, 4)
    pred_ys = pred_kpts[:, 1::2]  # (N, 4)
    target_xs = target_kpts[:, 0::2]
    target_ys = target_kpts[:, 1::2]

    # Compute bounding boxes (xyxy format)
    pred_x1, pred_y1 = pred_xs.min(1)[0], pred_ys.min(1)[0]
    pred_x2, pred_y2 = pred_xs.max(1)[0], pred_ys.max(1)[0]
    target_x1, target_y1 = target_xs.min(1)[0], target_ys.min(1)[0]
    target_x2, target_y2 = target_xs.max(1)[0], target_ys.max(1)[0]

    # Intersection
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area + 1e-7

    # IoU
    iou = inter_area / union_area

    # Enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1) + 1e-7

    # GIoU
    giou = iou - (enclose_area - union_area) / enclose_area

    return (1 - giou).mean()


def smooth_BCE(eps=0.1):
    """Returns label smoothing BCE targets for reducing overfitting; pos: `1.0 - 0.5*eps`, neg: `0.5*eps`. For details
    see https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441.
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """Modified BCEWithLogitsLoss to reduce missing label effects in YOLOv5 training with optional alpha smoothing."""

    def __init__(self, alpha=0.05):
        """Initializes a modified BCEWithLogitsLoss with reduced missing label effects, taking optional alpha smoothing
        parameter.
        """
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        """Computes modified BCE loss for YOLOv5 with reduced missing label effects, taking pred and true tensors,
        returns mean loss.
        """
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """Applies focal loss to address class imbalance by modifying BCEWithLogitsLoss with gamma and alpha parameters."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes FocalLoss with specified loss function, gamma, and alpha values; modifies loss reduction to
        'none'.
        """
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Calculates the focal loss between predicted and true labels using a modified BCEWithLogitsLoss."""
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """Implements Quality Focal Loss to address class imbalance by modulating loss based on prediction confidence."""

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        """Initializes Quality Focal Loss with given loss function, gamma, alpha; modifies reduction to 'none'."""
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        """Computes the focal loss between `pred` and `true` using BCEWithLogitsLoss, adjusting for imbalance with
        `gamma` and `alpha`.
        """
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    """Computes the total loss for YOLOv5 model predictions, including classification, box, and objectness losses."""

    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, _anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            if n := b.shape[0]:
                pkpts, _, pcls = pi[b, a, gj, gi].split((8, 1, self.nc), 1)  # keypoints, obj, cls

                # Regression (keypoints)
                pkpts = pkpts.sigmoid() * 2 - 0.5  # decode keypoints relative to grid cell

                # Keypoint regression loss: Smooth L1 + GIoU for shape alignment
                # L1 损失用于关键点坐标精度，GIoU损失用于整体形状
                loss_l1 = nn.functional.smooth_l1_loss(pkpts, tbox[i], reduction="mean", beta=0.5)
                loss_giou = polygon_giou_loss(pkpts, tbox[i])
                lbox += loss_l1 + 1.0 * loss_giou  # 提高GIoU权重从0.5到1.0

                # Objectness
                tobj[b, a, gj, gi] = 1.0  # positive samples get obj target = 1.0

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 8], tobj)  # objectness is at index 8 now
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,class,x1,y1,x2,y2,x3,y3,x4,y4) for loss computation."""
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # targets columns: (img_idx, cls, x1, y1, x2, y2, x3, y3, x4, y4) - 10列
        ncol = targets.shape[1]  # 10 for 四点格式
        gain = torch.ones(ncol + 1, device=self.device)  # +1 for anchor index appended later
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                ],
                device=self.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            # scale all keypoints to grid space: x,y repeated for 4 keypoints
            gain[2:10] = torch.tensor(shape)[[3, 2, 3, 2, 3, 2, 3, 2]]

            # Match targets to anchors
            t = targets * gain  # shape(na,nt,ncol+1)
            if nt:
                # 用4个关键点的外接框来匹配anchor
                kpts = t[..., 2:10]  # (na, nt, 8)
                xs = kpts[..., 0::2]  # x coords
                ys = kpts[..., 1::2]  # y coords
                bbox_w = xs.max(-1)[0] - xs.min(-1)[0]  # 外接框宽度
                bbox_h = ys.max(-1)[0] - ys.min(-1)[0]  # 外接框高度
                bbox_wh = torch.stack([bbox_w, bbox_h], dim=-1)  # (na, nt, 2)

                # Matches - 用外接框的wh与anchor匹配
                r = bbox_wh / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare
                t = t[j]  # filter

                # 用外接框中心作为grid分配依据
                kpts_filtered = t[:, 2:10]
                xs_f = kpts_filtered[:, 0::2]
                ys_f = kpts_filtered[:, 1::2]
                cx = (xs_f.min(-1)[0] + xs_f.max(-1)[0]) / 2
                cy = (ys_f.min(-1)[0] + ys_f.max(-1)[0]) / 2
                gxy = torch.stack([cx, cy], dim=-1)  # 外接框中心

                # Offsets
                gxi = gain[[2, 3]] - gxy  # inverse
                jj, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                jj = torch.stack((torch.ones_like(jj), jj, k, l, m))
                t = t.repeat((5, 1, 1))[jj]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[jj]

                # 重新计算gxy
                kpts_t = t[:, 2:10]
                xs_t = kpts_t[:, 0::2]
                ys_t = kpts_t[:, 1::2]
                cx_t = (xs_t.min(-1)[0] + xs_t.max(-1)[0]) / 2
                cy_t = (ys_t.min(-1)[0] + ys_t.max(-1)[0]) / 2
                gxy = torch.stack([cx_t, cy_t], dim=-1)
            else:
                t = targets[0]
                offsets = 0
                kpts_t = t[:, 2:10]
                xs_t = kpts_t[:, 0::2]
                ys_t = kpts_t[:, 1::2]
                cx_t = (xs_t.min(-1)[0] + xs_t.max(-1)[0]) / 2
                cy_t = (ys_t.min(-1)[0] + ys_t.max(-1)[0]) / 2
                gxy = torch.stack([cx_t, cy_t], dim=-1)

            # Define
            bc = t[:, :2]  # (image_idx, class)
            a = t[:, -1].long()  # anchor index (last column)
            b, c = bc.long().T  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))

            # Keypoint targets relative to grid cell
            gkpts = t[:, 2:10]  # keypoints in grid space
            gkpts_rel = gkpts.clone()
            for p_idx in range(4):
                gkpts_rel[:, p_idx * 2] -= gij[:, 0].float()  # relative to grid_x
                gkpts_rel[:, p_idx * 2 + 1] -= gij[:, 1].float()  # relative to grid_y
            tbox.append(gkpts_rel)

            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
