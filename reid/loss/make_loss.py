"""
Identity losses adapted from fast-reid (https://github.com/JDAI-CV/fast-reid), by liaoxingyu.
"""

import torch
import torch.nn.functional as F


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    return dist.clamp(min=1e-12).sqrt()


def triplet_loss(embedding, targets, anchor_weights):
    """Batch-hard triplet loss with soft margin (Hermans et al.), each anchor weighted by w_a (Eq. 5)."""
    dist_mat = euclidean_dist(embedding, embedding)
    n = dist_mat.size(0)
    is_pos = targets.view(n, 1).expand(n, n).eq(targets.view(n, 1).expand(n, n).t()).float()
    is_neg = 1 - is_pos

    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    y = torch.ones_like(dist_an)
    loss = F.soft_margin_loss(dist_an - dist_ap, y, reduction="none")
    loss = (anchor_weights * loss).sum() / anchor_weights.sum().clamp_min(1e-6)
    if loss == float("Inf"):
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
    return loss


def visibility_attention_loss(cls_attn, patch_visibility, synthetic, eps=1e-6):
    """L_vis (Eq. 4), on synthetic views only."""
    if not synthetic.any():
        return cls_attn.new_zeros((), dtype=torch.float32)
    attn = cls_attn[synthetic].float().mean(1)
    attn = attn / attn.sum(1, keepdim=True)
    target = patch_visibility[synthetic] / (eps + patch_visibility[synthetic].sum(1, keepdim=True))
    return -(target * attn.clamp_min(1e-8).log()).sum(1).mean()


def cross_entropy_loss(pred_class_outputs, gt_classes):
    log_probs = F.log_softmax(pred_class_outputs, dim=1)
    loss = -log_probs.gather(1, gt_classes.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)
    return loss.sum() / non_zero_cnt


def loss_func(outputs, gt_labels, domain_labels, patch_visibility, cfg):
    """L_total = lambda_id * L_id + lambda_tri * L_tri^w + lambda_dom * L_dom + lambda_vis * L_vis (Eq. 6)."""
    synthetic = domain_labels == 1
    # w_a, mean patch visibility of a synthetic view, 1 for a real image
    anchor_weights = torch.where(synthetic, patch_visibility.mean(1), torch.ones_like(patch_visibility[:, 0]))
    losses = {
        "ID_LOSS": cross_entropy_loss(outputs["cls_outputs"], gt_labels) * cfg.ID_LOSS_WEIGHT,
        "TRI_LOSS": triplet_loss(outputs["features"], gt_labels, anchor_weights) * cfg.TRIPLET_LOSS_WEIGHT,
        "VIS_LOSS": visibility_attention_loss(outputs["cls_attn"], patch_visibility, synthetic) * cfg.VIS_LOSS_WEIGHT,
    }
    if cfg.DOMAIN_LOSS_WEIGHT > 0:
        losses["DOMAIN_LOSS"] = F.cross_entropy(outputs["domain_logits"], domain_labels) * cfg.DOMAIN_LOSS_WEIGHT
    losses["TOTAL_LOSS"] = sum(losses.values())
    return losses
