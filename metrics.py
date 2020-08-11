"""Metrics

This file is for metrics code.
"""

import os
import torch
import time
import numpy as np
from collections import defaultdict

from loss import get_loss
from utils import get_index_to_incident_mapping, get_index_to_place_mapping

main_path = os.path.dirname(os.path.abspath(__file__))

# TODO: make these passed in
index_to_incident_mapping = get_index_to_incident_mapping()
index_to_place_mapping = get_index_to_place_mapping()


# TODO: move these out


def get_place_name_from_mapping(idx):
    name = None
    if idx in index_to_place_mapping:
        name = index_to_place_mapping[idx]
    else:
        name = "no place"
    return name


def get_incident_name_from_mapping(idx):
    name = None
    if idx in index_to_incident_mapping:
        name = index_to_incident_mapping[idx]
    else:
        name = "no incident"
    return name


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

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


def batched_index_select(input_value, dim, index):
    # TODO: confirm this code is correct w/ test cases
    """returns values from indices and along dim
    source: https://discuss.pytorch.org/t/batched-index-select/9115/11
    """
    for ii in range(1, len(input_value.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input_value.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input_value, dim, index)


def accuracy(output, target, topk=1):
    """Computes the topk accuracy and return the percentage.
    There must be some positive classes, otherwise an error is asserted.

    Args:
        output (Tensor): predicted probabilities, higher = more confidence
        target (Tensor): should be just 0s and 1s, with 1s being the positive class
        topk (int): number of top k elements to consider

    Returns:
        float: topk accuracy
    """
    probs, indices = torch.topk(output, topk)
    # extract values from target at topk indices
    index_select_output = batched_index_select(target, 1, indices)

    # TODO: need to handle case w/ multiple +1 labels, as this double counts
    correct_topk = index_select_output.view(-1).float().sum(0)
    num_pos_in_batch = target.view(-1).float().sum(0)
    if num_pos_in_batch == 0:
        # print("no pos in batch")
        return 100.0
        # raise ValueError("No positive classes (1) when computing accuracy.")
    # TODO: raise an error when more than one positive classes per batch dim
    return correct_topk.mul_(100.0 / num_pos_in_batch)


def get_acc_num_correct_out_of_total(output, target, topk=1):
    """Computes the topk accuracy and return the percentage.
    There must be some positive classes, otherwise an error is asserted.

    Args:
        output (Tensor): predicted probabilities, higher = more confidence
        target (Tensor): should be just 0s and 1s, with 1s being the positive class
        topk (int): number of top k elements to consider

    Returns:
        float: topk accuracy
    """
    probs, indices = torch.topk(output, topk)
    # extract values from target at topk indices
    index_select_output = batched_index_select(target, 1, indices)

    # TODO: need to handle case w/ multiple +1 labels, as this double counts
    correct_topk = index_select_output.view(-1).float().sum(0)
    num_pos_in_batch = target.view(-1).float().sum(0)
    return correct_topk, num_pos_in_batch


def validate(args, val_loader, all_models, epoch=None, writer=None):
    """Run validation of the model with metrics.

    Args:
        args:
        val_loader:
        all_models:

    Returns:
        float: incident mAP + place mAP
    """
    if epoch is None:
        raise NotImplementedError(
            "Not implemented for epoch==None")

    for m in all_models:
        # switch to evaluation mode
        m.eval()
    (trunk_model, incident_model, place_model) = all_models
    # holds the metrics
    a_v_batch_time = AverageMeter()
    a_v_data_time = AverageMeter()
    a_v_losses = AverageMeter()
    a_v_incident_top1 = AverageMeter()
    a_v_place_top1 = AverageMeter()
    a_v_incident_top5 = AverageMeter()
    a_v_place_top5 = AverageMeter()

    top1_num_correct_all, top1_num_total_all = 0, 0
    top5_num_correct_all, top5_num_total_all = 0, 0

    if args.activation == "softmax":
        # in this case, include "no incident" and "no place"
        ap_incidents = [[] for i in range(len(index_to_incident_mapping) + 1)]
        ap_places = [[] for i in range(len(index_to_place_mapping) + 1)]
    elif args.activation == "sigmoid":
        ap_incidents = [[] for i in range(len(index_to_incident_mapping))]
        ap_places = [[] for i in range(len(index_to_place_mapping))]

    # set end time as current time before training on a batch
    end_time = time.time()

    for batch_iteration, val_data_input in enumerate(val_loader):

        image_v = val_data_input[0].cuda(non_blocking=True)  # image variable (batch)
        target_p_v = val_data_input[1].cuda(non_blocking=True)  # p for place
        target_i_v = val_data_input[2].cuda(non_blocking=True)  # i for incident
        weight_p_v = val_data_input[3].cuda(non_blocking=True)
        weight_i_v = val_data_input[4].cuda(non_blocking=True)

        # measure data loading time
        a_v_data_time.update(time.time() - end_time)

        # compute output
        output = trunk_model(image_v)
        place_output = place_model(output)
        incident_output = incident_model(output)

        # get the loss
        loss, incident_output, place_output = get_loss(args,
                                                       incident_output,
                                                       target_i_v,
                                                       weight_i_v,
                                                       place_output,
                                                       target_p_v,
                                                       weight_p_v)

        # prepare for average precison calculations
        # make sure this is batch size
        assert incident_output.shape[0] == place_output.shape[0]
        for batch_idx in range(incident_output.shape[0]):
            np_incident_output = incident_output[batch_idx].cpu(
            ).detach().numpy()
            np_target_i_v = target_i_v[batch_idx].cpu().detach().numpy()
            np_weight_i_v = weight_i_v[batch_idx].cpu().detach().numpy()

            np_incident_output_shape = np_incident_output.shape[0]
            if args.activation == "softmax":
                np_incident_output_shape -= 1

            for class_idx in range(np_incident_output_shape):
                confidence = np_incident_output[class_idx]
                label = np_target_i_v[class_idx]
                weight = np_weight_i_v[class_idx]

                pos = (label == 1 and weight > 0)
                neg = (label == 0 and weight > 0)
                if pos:
                    ap_incidents[class_idx].append((confidence, 1))
                elif neg:
                    ap_incidents[class_idx].append((confidence, 0))

            np_place_output = place_output[batch_idx].cpu().detach().numpy()
            np_target_p_v = target_p_v[batch_idx].cpu().detach().numpy()
            np_weight_p_v = weight_p_v[batch_idx].cpu().detach().numpy()

            np_place_output_shape = np_place_output.shape[0]
            if args.activation == "softmax":
                np_place_output_shape -= 1

            for class_idx in range(np_place_output_shape):
                confidence = np_place_output[class_idx]
                label = np_target_p_v[class_idx]
                weight = np_weight_p_v[class_idx]

                pos = (label == 1 and weight > 0)
                neg = (label == 0 and weight > 0)
                if pos:
                    ap_places[class_idx].append((confidence, 1))
                elif neg:
                    ap_places[class_idx].append((confidence, 0))

        # incident accuracy
        incident_prec1 = accuracy(incident_output.data, target_i_v, topk=1)
        incident_prec5 = accuracy(incident_output.data, target_i_v, topk=5)

        top1_num_correct, top1_num_total = get_acc_num_correct_out_of_total(incident_output.data, target_i_v, topk=1)
        top1_num_correct_all += top1_num_correct
        top1_num_total_all += top1_num_total
        top5_num_correct, top5_num_total = get_acc_num_correct_out_of_total(incident_output.data, target_i_v, topk=5)
        top5_num_correct_all += top5_num_correct
        top5_num_total_all += top5_num_total

        # place accuracy
        place_prec1 = accuracy(place_output.data, target_p_v, topk=1)
        place_prec5 = accuracy(place_output.data, target_p_v, topk=5)

        a_v_losses.update(loss.data, image_v.size(0))
        a_v_place_top1.update(place_prec1, image_v.size(0))
        a_v_incident_top1.update(incident_prec1, image_v.size(0))
        a_v_place_top5.update(place_prec5, image_v.size(0))
        a_v_incident_top5.update(incident_prec5, image_v.size(0))

        # measure elapsed time
        a_v_batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_iteration % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {a_v_batch_time.val:.3f} ({a_v_batch_time.avg:.3f})\t'
                  'Data {a_v_data_time.val:.3f} ({a_v_data_time.avg:.3f})\t'
                  'Loss {a_v_losses.val:.4f} ({a_v_losses.avg:.4f})\t'
                  'Incident Prec@1 {a_v_incident_top1.val:.3f} ({a_v_incident_top1.avg:.3f})\t'
                  'Place Prec@1 {a_v_place_top1.val:.3f} ({a_v_place_top1.avg:.3f})\t'
                  'Place Prec@5 {a_v_place_top5.val:.3f} ({a_v_place_top5.avg:.3f})\t'
                  'Incident Prec@5 {a_v_incident_top5.val:.3f} ({a_v_incident_top5.avg:.3f})\t'.format(
                epoch, batch_iteration,
                len(val_loader),
                a_v_batch_time=a_v_batch_time,
                a_v_data_time=a_v_data_time,
                a_v_losses=a_v_losses,
                a_v_incident_top1=a_v_incident_top1,
                a_v_place_top1=a_v_place_top1,
                a_v_incident_top5=a_v_incident_top5,
                a_v_place_top5=a_v_place_top5))

    print("\nCalculating APs\n")
    # threshold are [0.0, 0.1, ..., 1.0] (11 values)
    thresholds = [round(i, 2) for i in list(np.linspace(0.0, 1.0, num=11))]

    # holds average precision for each class
    ap_incident_dict = {}
    ap_place_dict = {}

    # ap for incidents
    for i in range(len(ap_incidents)):
        class_points = ap_incidents[i]
        name = get_incident_name_from_mapping(i)
        if len(class_points) == 0:
            print("{} has no relevant labels".format(name))
            ap_incident_dict[name] = 1
            continue

        sorted_by_confidence = sorted(
            class_points, key=lambda x: x[0], reverse=True)

        count = 0
        pos_targets = 0
        max_prec = defaultdict(int)
        total_positives = int(np.sum(np.array(class_points)[:, 1]))
        if total_positives == 0:
            print("{} has no pos labels".format(name))
            continue  # alert in this case maybe

        # go in order
        for confidence, label in sorted_by_confidence:
            count += 1
            if label == 1:
                pos_targets += 1
            precision = pos_targets / count
            recall = pos_targets / total_positives

            for thresh in thresholds:
                if recall >= thresh:
                    max_prec[thresh] = max(max_prec[thresh], precision)
            if pos_targets == total_positives:
                break
        l = list(max_prec.values())
        average_precision = sum(l) / len(l)
        ap_incident_dict[get_incident_name_from_mapping(i)] = average_precision

    # repeat for places
    for i in range(len(ap_places)):
        class_points = ap_places[i]
        name = get_place_name_from_mapping(i)
        if len(class_points) == 0:
            print("{} has no relevant labels".format(name))
            ap_place_dict[name] = 1
            continue

        sorted_by_confidence = sorted(
            class_points, key=lambda x: x[0], reverse=True)

        count = 0
        pos_targets = 0
        max_prec = defaultdict(int)
        total_positives = int(np.sum(np.array(class_points)[:, 1]))
        if total_positives == 0:
            print("{} has no pos labels".format(name))
            continue  # alert in this case maybe

        # go in order
        for confidence, label in sorted_by_confidence:
            count += 1
            if label == 1:
                pos_targets += 1
            precision = pos_targets / count
            recall = pos_targets / total_positives
            for thresh in thresholds:
                if recall >= thresh:
                    max_prec[thresh] = max(max_prec[thresh], precision)

            if pos_targets == total_positives:
                break
        l = list(max_prec.values())
        average_precision = sum(l) / len(l)
        ap_place_dict[get_place_name_from_mapping(i)] = average_precision

    # TODO(ethan): move this code out for test set only
    if writer is None:  # for test mode
        import pickle
        import os
        incident_filename = os.path.basename(args.config) + "_incident_ap.pkl"
        place_filename = os.path.basename(args.config) + "_place_ap.pkl"
        pickle.dump(ap_incident_dict, open(incident_filename, "wb"))
        pickle.dump(ap_place_dict, open(place_filename, "wb"))

    # ap metrics
    incident_map = 0
    for incident, ap in ap_incident_dict.items():
        incident_map += ap
        if writer:
            writer.add_scalar('AP/incidents/{}'.format(incident), ap, epoch)
    incident_map /= len(ap_incident_dict)
    place_map = 0
    for place, ap in ap_place_dict.items():
        place_map += ap
        if writer:
            writer.add_scalar('AP/places/{}'.format(place), ap, epoch)
    place_map /= len(ap_place_dict)

    # show map
    if epoch is not None and writer is not None:
        writer.add_scalar('Loss/val', a_v_losses.avg, epoch)
        writer.add_scalar('Accuracy/val_place_1',
                          a_v_place_top1.avg, epoch)
        writer.add_scalar('Accuracy/val_place_5',
                          a_v_place_top5.avg, epoch)
        writer.add_scalar('Accuracy/val_incident_1',
                          a_v_incident_top1.avg, epoch)
        writer.add_scalar('Accuracy/val_incident_5',
                          a_v_incident_top5.avg, epoch)
        writer.add_scalar('mAP/incidents', incident_map, epoch)
        writer.add_scalar('mAP/places', place_map, epoch)

    print("incident map", incident_map)
    print("place map", place_map)
    print("incident top1", top1_num_correct_all / top1_num_total_all)
    print("incident top5", top5_num_correct_all / top5_num_total_all)
    return incident_map + place_map
