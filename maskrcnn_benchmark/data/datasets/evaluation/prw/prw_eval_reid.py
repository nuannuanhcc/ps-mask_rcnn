
from __future__ import division

import torch
import os
import time
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import logging

import pandas as pd
import os.path as osp
from sklearn.metrics import average_precision_score, precision_recall_curve

from IPython import embed

def do_prw_evaluation_reid(datasets, predictions, output_folder, logger):
    ###########################################################
    tq_flag = False
    if isinstance(datasets, (list, tuple)):
        assert len(datasets) == 2, "dataset type {} error!".format(type(datasets))
        dataset_test, dataset_query = datasets
        predictions_test, predictions_query = predictions
        query_img_list = dataset_query.ids
        tq_flag = True
    else:
        dataset_test = datasets
        predictions_test = predictions

    output_folder = output_folder
    logger = logger

    dataset = dataset_test
    predictions = predictions_test
    ############################################################

    # TODO need to make the use_07_metric format available
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []
    #test_img_list = dataset.ids # re_id
    test_img_list = [] #
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)

        if len(prediction) == 0:
            continue

        ### re_id
        img_name = dataset.ids[image_id]  # re_id
        test_img_list.append(img_name)  # re_id
        ###

        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        pred_boxlists.append(prediction)

        gt_boxlist = dataset.get_groundtruth(image_id)
        gt_boxlists.append(gt_boxlist)

    ######################################################



    print ('do_prw_evaluation ...')
    #embed() ###

    if tq_flag:
        result = eval_reid_prw(
            test_img_list=test_img_list,
            query_img_list=query_img_list,
            query_boxlists=predictions_query,
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=0.5,
            use_07_metric=False,
        )
    elif not tq_flag:
        result = eval_detection_prw(
            pred_boxlists=pred_boxlists,
            gt_boxlists=gt_boxlists,
            iou_thresh=0.5,
            use_07_metric=False,
        )
    else:
        raise KeyError(tq_flag)

    """
    result = eval_detection_prw(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=False,
    )
    """

    ### re_id
    """
    reid_result = eval_reid_prw(
        test_img_list=test_img_list,
        query_img_list=query_img_list,
        query_boxlists=predictions_query,
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=0.5,
        use_07_metric=False,
    )
    """

    if tq_flag:
        topk = [1, 3, 5, 10]
        result_str = "##################################################"
        result_str += '\nmodel: {}\n'.format(result['model'])
        result_str += "Detection_Recall: {:.4f}\n".format(result["Detection_Recall"])
        result_str += "Detection_Precision: {:.4f}\n".format(result["Detection_Precision"])
        result_str += "Detection_mean_Avg_Precision: {:.4f}\n".format(result["Detection_mean_Avg_Precision"])
        result_str += "ReID_Recall: {:.4f}  ".format(result["ReID_Recall"])
        result_str += "(ReID_Recall_Ideal: {:.4f})\n".format(result["ReID_Recall_Ideal"])
        result_str += "ReID_mean_Avg_Precision: {:.4f}  ".format(result["ReID_mean_Avg_Precision"])
        result_str += "(ReID_mean_Avg_Precision_Ideal: {:.4f})\n".format(result["ReID_mean_Avg_Precision_Ideal"])
        for i, k in enumerate(topk):
            result_str += '  Top-{:2d} = {:.2%} (The ideal top-{:2d} = {:.2%})\n'.format(k, result["CMC"][i], k, result["CMC_Ideal"][i])
        result_str += "##################################################"
    elif not tq_flag:
        result_str = "##################################################"
        for i, ap in enumerate(result["ap"]):
            if i == 0:  # skip background
                continue
            result_str += "{:<16}: {:.4f}\n".format(
                dataset.map_class_id_to_class_name(i), ap
            )
        result_str += "##################################################"

    """
    for i, ap in enumerate(result["ap"]):
        if i == 0:  # skip background
            continue
        result_str += "{:<16}: {:.4f}\n".format(
            dataset.map_class_id_to_class_name(i), ap
        )
    """
    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "result.txt"), "a") as fid:
            fid.write(result_str)
    return result


def eval_detection_prw(pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."
    prec, rec = calc_detection_prw_prec_rec(pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh)
    #print ('Precision: {}'.format(np.nanmean(prec)))
    #print ('Recall: {}'.format(np.nanmean(rec)))
    print ('Precision: {}'.format(prec))
    print ('Recall: {}'.format(rec))
    ap = calc_detection_prw_ap(prec, rec, use_07_metric=use_07_metric)
    return {"ap": ap, "map": np.nanmean(ap)}


def calc_detection_prw_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            #print ('For calc_detection_prw_prec_rec 1 ...')
            #embed() ###

            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1] # sorted
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        #print ('For calc_detection_prw_prec_rec 2 ...')
        #embed()  ###

        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp) # precision / accuracy
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l] # recall

    return prec, rec


def calc_detection_prw_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

############################################  Re-ID evaluation   ##################################################

def eval_reid_prw(test_img_list, query_img_list, query_boxlists, pred_boxlists, gt_boxlists, iou_thresh=0.5, use_07_metric=False):
    """Evaluate on voc dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
        use_07_metric: boolean
    Returns:
        dict represents the results
    """
    #assert len(test_img_list) == 6978, "Length of test_img lists need to be 6978, but it is {}.".format(len(test_img_list))
    assert len(query_img_list) == 2900, "Length of query_img lists need to be 2900, but it is {}.".format(len(query_img_list))
    assert len(query_boxlists) == 2900, "Length of query_boxlists lists need to be 2900, but it is {}.".format(len(query_boxlists))
    assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."

    result = {}
    result.update({'model': 'model_0'})

    prec, rec = calc_reid_prw_prec_rec(pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh)
    print('Person Detection:')
    Detection_Precision = 100 * np.nanmean(prec[1])
    Detection_Recall = 100 * np.nanmean(rec[1])
    print ('Precision: {}%'.format(Detection_Precision))
    print ('Recall: {}%'.format(Detection_Recall))
    result.update({'Detection_Precision': Detection_Precision})
    result.update({'Detection_Recall': Detection_Recall})
    #print ('Precision: {}'.format(np.nanmean(prec)))
    #print ('Recall: {}'.format(np.nanmean(rec)))
    ap = calc_reid_prw_ap(prec, rec, use_07_metric=use_07_metric)
    Detection_mean_Avg_Precision = np.nanmean(ap)
    print('  mAP = {:.2%}'.format(Detection_mean_Avg_Precision))
    result.update({'Detection_mean_Avg_Precision': Detection_mean_Avg_Precision})

    ##################  Re-ID   ###################
    topk = [1, 3, 5, 10]
    #CMC = []
    real_result, ideal_result = calc_reid_prw_topk(test_img_list, query_img_list, query_boxlists, pred_boxlists, gt_boxlists, topk, det_thresh=0.5, iou_thresh=0.5)
    rec, aps, accs = real_result
    rec_ideal, aps_ideal, accs_ideal = ideal_result
    print('Person Search ranking:')
    ReID_Recall = np.nanmean(rec)
    ReID_Recall_Ideal = np.nanmean(rec_ideal)
    ReID_mean_Avg_Precision = np.nanmean(aps)
    ReID_mean_Avg_Precision_Ideal = np.nanmean(aps_ideal)
    print('  recall = {:.2%} (The ideal recall = {:.2%})'.format(ReID_Recall, ReID_Recall_Ideal))
    print('  mAP = {:.2%} (The ideal mAP = {:.2%})'.format(ReID_mean_Avg_Precision, ReID_mean_Avg_Precision_Ideal))
    result.update({'ReID_Recall': ReID_Recall})
    result.update({'ReID_Recall_Ideal': ReID_Recall_Ideal})
    result.update({'ReID_mean_Avg_Precision': ReID_mean_Avg_Precision})
    result.update({'ReID_mean_Avg_Precision_Ideal': ReID_mean_Avg_Precision_Ideal})
    accs = np.mean(accs, axis=0)
    accs_ideal = np.mean(accs_ideal, axis=0)
    for i, k in enumerate(topk):
        print('  top-{:2d} = {:.2%} (The ideal top-{:2d} = {:.2%})'.format(k, accs[i], k, accs_ideal[i]))
    result.update({'CMC': accs})
    result.update({'CMC_Ideal': accs_ideal})
    #return {"ap": ap, "map": np.nanmean(ap), "CMC": accs}
    return result

def calc_reid_prw_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)
    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()
            gt_index = iou.argmax(axis=1)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp) # precision / accuracy
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l] # recall

    return prec, rec


def calc_reid_prw_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.
    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0.0, 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def calc_reid_prw_topk(test_img_list, query_img_list, query_boxlists, pred_boxlists, gt_boxlists, topk, det_thresh=0.5, iou_thresh=0.5, gallery_size=100):

    #assert len(test_img_list) == 6978, "Length of test_img lists need to be 6978, but it is {}.".format(len(test_img_list))
    assert len(query_img_list) == 2900, "Length of query_img lists need to be 2900, but it is {}.".format(len(query_img_list))
    assert len(query_boxlists) == 2900, "Length of query_boxlists lists need to be 2900, but it is {}.".format(len(query_boxlists))
    assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."
    assert isinstance(topk, (list, tuple)), "topk must be a list or tuple !"

    iou_thresh = iou_thresh
    logger = logging.getLogger("prw_reid_benchmark.evaluation\n")
    logger.info("Start evaluating")
    logger.info("Detection iou_thresh: {}".format(iou_thresh))

    ############################# path #############################
    annotation_dir = '/home/person_search/ps-mask_rcnn/data/sysu/SIPN_annotation/'

    test_all_file = 'testAllDF.csv'
    query_file = 'queryDF.csv'
    q_to_g_file = 'q_to_g' + str(gallery_size) + 'DF.csv'

    test_all = pd.read_csv(osp.join(annotation_dir, test_all_file))
    query_boxes = pd.read_csv(osp.join(annotation_dir, query_file))
    queries_to_galleries = pd.read_csv(osp.join(annotation_dir, q_to_g_file))

    test_all, query_boxes = delta_to_coordinates(test_all, query_boxes)
    ################################ initial ################################
    test_imnames = test_img_list
    query_imnames = query_img_list

    #print ('calc_reid_prw_topk 1 ...')
    #embed()  ###

    #gallery_det_boxes = torch.cat([pred_boxlist.bbox for pred_boxlist in pred_boxlists])
    #gallery_det_scores = torch.cat([pred_boxlist.get_field('scores') for pred_boxlist in pred_boxlists]).reshape([-1, 1])
    #gallery_det = torch.cat((gallery_det_boxes, gallery_det_scores), dim=1)
    #gallery_feat = torch.cat([pred_boxlist.get_field('reid_feature') for pred_boxlist in pred_boxlists])
    #probe_feat = torch.cat([query_boxlist.get_field('reid_feature') for query_boxlist in query_boxlists])

    gallery_det_boxes = [pred_boxlist.bbox.numpy() for pred_boxlist in pred_boxlists]
    gallery_det_scores = [pred_boxlist.get_field('scores').numpy() for pred_boxlist in pred_boxlists]
    gallery_det_scores = [np.array(g_d_s).reshape([-1, 1]) for g_d_s in gallery_det_scores]
    gallery_det = [np.concatenate([g_d_b, g_d_s], axis=1) for g_d_b, g_d_s in zip(gallery_det_boxes, gallery_det_scores)]
    gallery_feat = [pred_boxlist.get_field('reid_feature').numpy() for pred_boxlist in pred_boxlists]
    probe_feat = [query_boxlist.get_field('reid_feature').numpy() for query_boxlist in query_boxlists]
    probe_feat = [p_f.squeeze() for p_f in probe_feat]

    #print ('calc_reid_prw_topk 2 ...')
    #embed() ###


    use_full_set = gallery_size == -1
    df = test_all.copy()

    # ====================formal=====================
    name_to_det_feat = {}
    for name, det, feat in zip(test_imnames, gallery_det, gallery_feat):
        scores = det[:, 4].ravel()
        inds = np.where(scores >= det_thresh)[0]
        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds], feat[inds])

    # # =====================debug=====================
    # f = open('name_to_det_feat.pkl', 'rb+')
    # name_to_det_feat = pickle.load(f)
    # # ======================end======================

    rec = []
    rec_ideal = []
    aps = []
    aps_ideal = [] ### Ideal situation
    accs = []
    accs_ideal = [] ### Ideal situation
    #topk = [1, 5, 10]
    topk = topk
    # ret  # TODO: save json
    for i in range(len(probe_feat)):
        #print ('calc_reid_prw_topk i = {} ...'.format(i))
        #embed()  ###

        pid = query_boxes.loc[i, 'pid']
        num_g = query_boxes.loc[i, 'num_g']
        y_true, y_score = [], []
        y_true_ideal = [] ### Ideal situation
        imgs, rois = [], []
        count_gt, count_tp = 0, 0
        count_tp_ideal = 0 ### Ideal situation
        # Get L2-normalized feature vector
        feat_p = probe_feat[i].ravel()
        # Ignore the probe image
        start = time.time()
        probe_imname = queries_to_galleries.iloc[i, 0]
        # probe_roi = df[df['imname'] == probe_imname]
        # probe_roi = probe_roi[probe_roi['is_query'] == 1]
        # probe_roi = probe_roi[probe_roi['pid'] == pid]
        # probe_roi = probe_roi.loc[:, 'x1': 'y2'].as_matrix()
        probe_gt = []
        tested = set([probe_imname])
        # 1. Go through the gallery samples defined by the protocol
        for g_i in range(1, gallery_size + 1):
            #print ('calc_reid_prw_topk g_i = {} ...'.format(g_i))
            #embed()  ###

            gallery_imname = queries_to_galleries.iloc[i, g_i]
            # gt = df[df['imname'] == gallery_imname]
            # gt = gt[gt['pid'] == pid]  # important
            # gt = gt.loc[:, 'x1': 'y2']
            if g_i <= num_g:
                gt = df.query('imname==@gallery_imname and pid==@pid')
                gt = gt.loc[:, 'x1': 'y2'].values.ravel() #################################
                #gt = gt.loc[:, 'x1': 'y2'].as_matrix().ravel()
            else:
                gt = np.array([])
            count_gt += (gt.size > 0)
            # compute distance between probe and gallery dets
            if gallery_imname not in name_to_det_feat: continue
            det, feat_g = name_to_det_feat[gallery_imname]
            # get L2-normalized feature matrix NxD
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2])
            # compute cosine similarities
            sim = feat_g.dot(feat_p).ravel()
            # assign label for each det
            label = np.zeros(len(sim), dtype=np.int32)
            label_ideal = np.zeros(len(sim), dtype=np.int32) ### Ideal situation
            if gt.size > 0:
                w, h = gt[2] - gt[0], gt[3] - gt[1]
                probe_gt.append({'img': str(gallery_imname), 'roi': list(gt.astype('float'))})
                iou_thresh = min(.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(sim)[::-1]
                sim = sim[inds]
                det = det[inds]

                label_ideal[0] = 1 # Ideally
                count_tp_ideal += 1 # Ideally
                #label[0] = 1
                #count_tp += 1
                #"""
                # only set the first matched det as true positive
                for j, roi in enumerate(det[:, :4]):
                    if _compute_iou(roi, gt) >= iou_thresh:
                        label[j] = 1
                        count_tp += 1
                        break
                #"""

            y_true.extend(list(label))
            y_true_ideal.extend(list(label_ideal)) ### Ideal situation
            y_score.extend(list(sim))
            imgs.extend([gallery_imname] * len(sim))
            rois.extend(list(det))
            tested.add(gallery_imname)
        # 2. Go through the remaining gallery images if using full set
        if use_full_set:
            pass  # TODO
        # 3. Compute AP for this probe (need to scale by recall rate)
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        y_true_ideal = np.asarray(y_true_ideal) ### Ideal situation
        assert count_tp <= count_gt
        assert count_tp_ideal <= count_gt ### Ideal situation
        if count_gt == 0:
            print(probe_imname, i)
            break
        recall_rate = count_tp * 1.0 / count_gt
        recall_rate_ideal = count_tp_ideal * 1.0 / count_gt ### Ideal situation
        ap = 0 if count_tp == 0 else average_precision_score(y_true, y_score) * recall_rate
        ap_ideal = 0 if count_tp_ideal == 0 else average_precision_score(y_true_ideal, y_score) * recall_rate_ideal ### Ideal situation
        rec.append(recall_rate)
        rec_ideal.append(recall_rate_ideal)
        aps.append(ap)
        aps_ideal.append(ap_ideal) ### Ideal situation
        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        y_true_ideal = y_true_ideal[inds] ### Ideal situation

        #print ('for query i = {} ...'.format(i))
        #embed()  ###

        """
        num = 0.0
        ap = []
        for i, y_t in enumerate(y_true):
            if y_t == 1:
                num += 1
                ap.append(num / (int(i) + 1))
        aps.append(np.mean(ap))
        """

        accs.append([min(1, sum(y_true[:k])) for k in topk])
        accs_ideal.append([min(1, sum(y_true_ideal[:k])) for k in topk]) ### Ideal situation
        # compute time cost
        end = time.time()
        #print('{}-th loop, cost {:.4f}s'.format(i, end - start)) ######################
    return (rec, aps, accs), (rec_ideal, aps_ideal, accs_ideal)


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def delta_to_coordinates(test_all, query_boxes):
    """change `del_x` and `del_y` to `x2` and `y2` for testing set"""
    test_all['del_x'] += test_all['x1']
    test_all['del_y'] += test_all['y1']
    test_all.rename(columns={'del_x': 'x2', 'del_y': 'y2'}, inplace=True)
    query_boxes['del_x'] += query_boxes['x1']
    query_boxes['del_y'] += query_boxes['y1']
    query_boxes.rename(columns={'del_x': 'x2', 'del_y': 'y2'}, inplace=True)
    return test_all, query_boxes



