import torch
import torch.nn.functional as F
from random import randint
from torch.utils.checkpoint import checkpoint


def jaccard_loss(pred, target):
    eps = 1e-9
    pred = pred.float()
    target = target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    ious = (intersection + eps) / (union - intersection + eps)
    ious = torch.clamp(ious, 0, 1)
    return ious.mean()


def hybrid_loss(pred, target, pred_sig=None):
    #assert not torch.isnan(pred).any()
    dt = pred.dtype
    if pred_sig is None:
        pred_sig = torch.sigmoid(pred).to(dt)
    bce = F.binary_cross_entropy_with_logits(pred.float(), target.float())
    pred = pred.to(dt)
    target = target.to(dt)
    bce = bce.to(dt)
    jl = jaccard_loss(pred_sig, target).to(dt)
    ljac = torch.log(jl).to(dt)
    return bce - ljac


def pack(mask, image, binarize=True):
    if binarize:
        return torch.cat(((mask > 0.5).float(), image), dim=1)
    return torch.cat((mask.float(), image), dim=1)


def discriminator_loss(fake_scores, real_scores):
    ones = torch.ones(real_scores.shape, device=real_scores.device, dtype=real_scores.dtype)
    zeros = torch.zeros(fake_scores.shape, device=fake_scores.device, dtype=real_scores.dtype)
    real_bce = F.binary_cross_entropy_with_logits(real_scores, ones)
    fake_bce = F.binary_cross_entropy_with_logits(fake_scores, zeros)
    return (real_bce + fake_bce)/2


def generator_loss(log_pred_mask, real_mask, pred_scores, adversarial_weight=0.5, pred_mask=None):
    adversarial_weight = float(adversarial_weight)
    if 0 < adversarial_weight < 1:
        ones = torch.ones(pred_scores.shape, device=pred_scores.device, dtype=pred_scores.dtype)
        adv_loss = F.binary_cross_entropy_with_logits(pred_scores, ones)
        hybrid_weight = 1-adversarial_weight
        return hybrid_weight*hybrid_loss(log_pred_mask, real_mask, pred_mask) + adversarial_weight*adv_loss
    elif adversarial_weight >= 1:
        ones = torch.ones(pred_scores.shape, device=pred_scores.device, dtype=pred_scores.dtype)
        adv_loss = F.binary_cross_entropy_with_logits(pred_scores, ones)
        return adv_loss
    else:
        return hybrid_loss(log_pred_mask, real_mask, pred_mask)


def conv_orthogonality_loss(model: torch.nn.Module, lmbda=0.0001):
    total_loss = 0
    count = 0
    for param in model.parameters():
        if len(param.shape) == 4:
            count += 1
            out, *_ = param.shape
            flat = param.reshape(out, -1)
            normal_matrix = flat@flat.transpose(0, 1)
            ortho_dev = normal_matrix-torch.eye(out, dtype=param.dtype,device=param.device)
            total_loss += ortho_dev.norm()
    return lmbda*total_loss/count
