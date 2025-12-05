import torch
import torch.nn.functional as F

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr


def structure_loss(pred, mask):
    """
    loss function (ref:F3Net-AAAI-2020)
    structure_loss函数接受两个参数pred和mask，分别代表模型的预测分割图和真实分割图。
    该函数返回模型预测值和真实值之间的结构损失，用于训练和优化模型。
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)  # 计算权重矩阵weit，用于加权WBCE和WIoU损失
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')  # 计算二元交叉熵损失，使用sigmoid将预测值映射到0-1之间
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))  # 加权WBCE损失

    pred = torch.sigmoid(pred)  # 使用sigmoid将预测值映射到0-1之间
    inter = ((pred * mask) * weit).sum(dim=(2, 3))  # 加权交集
    union = ((pred + mask) * weit).sum(dim=(2, 3))  # 加权并集
    wiou = 1 - (inter + 1) / (union - inter + 1)  # 加权交并比
    return (wbce + wiou).mean()  # 计算最终损失，将WBCE和WIoU损失相加求平均