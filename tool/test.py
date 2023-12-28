import os
import sys
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs 
from util.DSNetDataset import DSNetDataset 
from util.voxelize import voxelize
from util import transform as t
from util.data_util import collate_fn
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score

def report_scores(pred, gt):
    names = ['Prec.', 'Rec.', 'F1', 'mIoU']
    precision, recall, f1, _ = precision_recall_fscore_support(gt, pred)
    oa = accuracy_score(gt, pred)
    mIoU = jaccard_score(gt, pred, average=None)
    scores = [precision, recall, f1, mIoU]
    for name, score in zip(names, scores):
        print(f'\t{name}: \t{score}, Mean: \t{np.mean(score)}')
    print(f'OA: {oa}')

def get_parser():
    parser = argparse.ArgumentParser(description='DS-Net')
    parser.add_argument('--config', type=str, default='config/DS-Net/config.yaml', help='config file')
    parser.add_argument('opts', help='see config/DS-Net/config.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.num_classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.num_classes))


    if args.arch == 'scene_seg_net':
        from architectures import SceneSegNet as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(args).cuda()
    logger.info(model)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {}), (best score {})".format(args.model_path, checkpoint['epoch'], checkpoint['best_iou']))
        args.epoch = checkpoint['epoch']
        args.manual_seed = checkpoint['seed']
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))


    test_transform = None

    test_data = DSNetDataset(
        transform=test_transform,
        with_intensity=args.with_intensity,
        config=args,
        split=args.test_split
    )

    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=args.batch_size,
        shuffle=False, 
        num_workers=args.test_workers, 
        pin_memory=True, 
        sampler=None, 
        drop_last=False,
        collate_fn=collate_fn
    )
    test(model, test_loader, criterion)


def test(model, test_loader, criterion, n_vote=1):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    model.eval()

    check_makedirs(args.save_folder)
    
    
    softmax = nn.Softmax(dim=1)
    smoothing = 0.95
    best_iou = 0
    best_vote = 0

    for v in range(n_vote):

        prob_list, seg_list, cls_list, offset_list = [], [], [], []
         

        for i, (coord, feat, target, offset) in enumerate(test_loader):
            
            coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)

            with torch.no_grad():
                output = model(coord, feat, offset)

            output = softmax(output) # n, n_cls 

            prob_list.append(output.detach().cpu().numpy())
            seg_list.append(target.view(-1).detach().cpu().numpy())
            offset_list.append(offset.view(-1).detach().cpu().numpy())

        prob_list_each = []
        seg_list_each = []
        for b in range(len(prob_list)):
            start_ind = 0
            for o in offset_list[b]:
                end_ind = o
                prob_list_each.append(prob_list[b][start_ind:end_ind])
                seg_list_each.append(seg_list[b][start_ind:end_ind])
                start_ind = end_ind
        assert len(prob_list_each) == len(seg_list_each)
        if v == 0:
            averaged_probs = prob_list_each 
        else:
            N = len(averaged_probs)
            averaged_probs = [smoothing * averaged_probs[i] + (1 - smoothing) * prob_list_each[i] for i in range(N)] 

        current_pred = [np.argmax(x, axis=1).reshape(-1) for x in averaged_probs] 
        
        intersection, union, target = intersectionAndUnion(
            np.concatenate(current_pred, 0),
            np.concatenate(seg_list_each, 0),
            K=args.num_classes
        )
        IoU_class = intersection / (union + 1e-10) 
        mIoU = np.mean(IoU_class)

        logger.info(f'result of vote {v}: mIoU {mIoU:.4f}.')
        for c in range(args.num_classes):
            logger.info(f'IoU of class {c}: {IoU_class[c]:.4f}')

        if mIoU > best_iou:
            best_iou = mIoU 
            best_vote = v
            # saving best pred and label  
            
            with open(os.path.join(args.save_folder, 'pred.pkl'), 'wb') as f:
                pickle.dump(current_pred, f)
        if v == 0:
            with open(os.path.join(args.save_folder, 'seg.pkl'), 'wb') as f:
                pickle.dump(seg_list_each, f)
            
        logger.info(f'Best IoU in vote {best_vote}: {best_iou:.4f}')

    # after all voting 
    logger.info(f'voting finished. Best IoU is {best_iou:.4f}')
    logger.info('Printing scores ... ')
    report_scores(np.concatenate(current_pred, 0), np.concatenate(seg_list_each, 0))
            

if __name__ == '__main__':
    main()
