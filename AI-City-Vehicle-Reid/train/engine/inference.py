# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import torch
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, R1_mAP_reranking, Track_R1_mAP


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, tids = batch
            data = data.cuda()
            feat = model(data)
            return feat, pids, camids, tids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM), 'R1_mAP_reranking': R1_mAP_reranking(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM), 'Track_R1_mAP': Track_R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM), 'R1_mAP_reranking': R1_mAP_reranking(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM), 'Track_R1_mAP': Track_R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    re_cmc, re_mAP = evaluator.state.metrics['R1_mAP_reranking']
    track_cmc, track_mAP = evaluator.state.metrics['Track_R1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 100]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    logger.info("re_mAP: {:.1%}".format(re_mAP))
    for r in [1, 5, 10, 100]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, re_cmc[r - 1]))

    logger.info("track_mAP: {:.1%}".format(track_mAP))
    for r in [1, 5, 10, 100]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, track_cmc[r - 1]))