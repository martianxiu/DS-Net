import os

import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def Accuracy(pred_list, target_list, num_classes=2):
    pred = np.concatenate(pred_list).ravel() # b
    gt = np.concatenate(target_list).ravel() # b
    OA = np.sum(pred == gt) / (gt.shape[0])

    pred_cls_wise = [] 
    _, gt_cls_wise = np.unique(gt, return_counts=True)
    for c in range(num_classes):
        intersection = np.sum(np.logical_and(pred == c, gt == c))
        pred_cls_wise.append(intersection/gt_cls_wise[c]) 
    MA = np.mean(pred_cls_wise)
     
    return OA, MA, pred_cls_wise 
    
class CrossEntropyWithSmoothing(object):
    def __init__(self, label_smoothing=0.0, class_weight=None, ignore_index=None):
        self.eps = label_smoothing
        self.class_weight = class_weight

    def __call__(self, pred, gold):
        gold = gold.view(-1) # (batchsize, )

        if self.eps != 0.0:
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1) # (batch_size, num_class)
            if self.class_weight is not None:
                w = self.class_weight[gold].view(-1, 1)
                loss = - (w * one_hot * log_prb).sum(dim=1).mean()
            else:
                loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(pred, gold, weight=self.class_weight, reduction='mean')

        return loss

   

def cal_loss(pred, gold, weight=None, smoothing=False, eps=0.2, ignore_label=None):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    pred = pred.view(-1)
    gold = gold.contiguous().view(-1) # (batchsize, )

    if smoothing:
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1) # (batch_size, num_class)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, weight=weight, reduction='mean')

    return loss



def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target



def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port
