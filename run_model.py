"""run_model.py

This is the main executable file for running the IncidentsDataset code.
Training, validation, and testing of the models occurs from this entrypoint.

Helpful resources:
    - https://github.com/pytorch/examples/blob/master/imagenet/main.py
"""

from datetime import datetime
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import os
import pprint
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

cudnn.benchmark = True

from metrics import AverageMeter, accuracy, validate
import architectures as architectures
from loss import get_loss
from parser import get_parser, get_postprocessed_args
from dataset import get_dataset
from utils import save_checkpoint


def train(args, train_loader, all_models, optimizer, epoch):
    """
    Trains for one epoch of the train_loader dataset.
    """
    # switch all models to train mode
    for m in all_models:
        m.train()

    (trunk_model, incident_layer, place_layer) = all_models

    # holds some metrics
    a_v_batch_time = AverageMeter()
    a_v_data_time = AverageMeter()
    a_v_losses = AverageMeter()
    a_v_incident_top1 = AverageMeter()
    a_v_place_top1 = AverageMeter()
    a_v_incident_top5 = AverageMeter()
    a_v_place_top5 = AverageMeter()

    # set end time as current time before training on a batch
    end_time = time.time()

    for batch_iteration, (input_data, target_p_v, target_d_v, weight_p_v, weight_d_v) in enumerate(train_loader):
        # measure data loading time
        a_v_data_time.update(time.time() - end_time)

        image_v = input_data.cuda(non_blocking=True)
        target_p_v = target_p_v.cuda(non_blocking=True)
        target_d_v = target_d_v.cuda(non_blocking=True)
        weight_p_v = weight_p_v.cuda(non_blocking=True)
        weight_d_v = weight_d_v.cuda(non_blocking=True)

        # input_v = torch.autograd.Variable(image_v)
        # target_p_v = torch.autograd.Variable(target_p_v)
        # target_d_v = torch.autograd.Variable(target_d_v)
        # weight_p_v = torch.autograd.Variable(weight_p_v)
        # weight_d_v = torch.autograd.Variable(weight_d_v)

        # compute output
        output = trunk_model(image_v)
        place_output = place_layer(output)
        incident_output = incident_layer(output)

        # get the loss according to parameters
        loss, incident_output, place_output = get_loss(args,
                                                       incident_output,
                                                       target_d_v,
                                                       weight_d_v,
                                                       place_output,
                                                       target_p_v,
                                                       weight_p_v)

        # measure accuracy and record loss
        incident_prec1, incident_prec5 = accuracy(incident_output.data, target_d_v, topk=1), \
                                         accuracy(incident_output.data, target_d_v, topk=5)
        place_prec1, place_prec5 = accuracy(place_output.data, target_p_v, topk=1), \
                                   accuracy(place_output.data, target_p_v, topk=5)
        a_v_losses.update(loss.data, input_data.size(0))
        a_v_place_top1.update(place_prec1, input_data.size(0))
        a_v_incident_top1.update(incident_prec1, input_data.size(0))
        a_v_place_top5.update(place_prec5, input_data.size(0))
        a_v_incident_top5.update(incident_prec5, input_data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                len(train_loader),
                a_v_batch_time=a_v_batch_time,
                a_v_data_time=a_v_data_time,
                a_v_losses=a_v_losses,
                a_v_incident_top1=a_v_incident_top1,
                a_v_place_top1=a_v_place_top1,
                a_v_incident_top5=a_v_incident_top5,
                a_v_place_top5=a_v_place_top5))
        # TODO: add more metrics here
        writer.add_scalar('Loss/train', a_v_losses.avg,
                          batch_iteration + epoch * len(train_loader))
        writer.add_scalar('Accuracy/train_place_1', a_v_place_top1.avg,
                          batch_iteration + epoch * len(train_loader))
        writer.add_scalar('Accuracy/train_place_5', a_v_place_top5.avg,
                          batch_iteration + epoch * len(train_loader))
        writer.add_scalar('Accuracy/train_incident_1', a_v_incident_top1.avg,
                          batch_iteration + epoch * len(train_loader))
        writer.add_scalar('Accuracy/train_incident_5', a_v_incident_top5.avg,
                          batch_iteration + epoch * len(train_loader))


# global variables
best_mean_ap = None
parser = get_parser()
writer = None


def main():
    global best_mean_ap, parser, writer
    args = parser.parse_args()
    args = get_postprocessed_args(args)

    print("args: \n")
    pprint.pprint(args)

    # create the model
    print("creating model with feature trunk architecture: '{}'".format(args.arch))

    # the shared feature trunk model
    trunk_model = architectures.get_trunk_model(args)
    # the incident model
    incident_layer = architectures.get_incident_layer(args)
    # the place model
    place_layer = architectures.get_place_layer(args)

    print("parallelizing models with {} gpus".format(args.num_gpus))
    trunk_model = nn.DataParallel(
        trunk_model,
        device_ids=range(args.num_gpus)
    ).cuda()
    incident_layer = nn.DataParallel(
        incident_layer,
        device_ids=range(args.num_gpus)
    ).cuda()
    place_layer = nn.DataParallel(
        place_layer,
        device_ids=range(args.num_gpus)
    ).cuda()

    if args.checkpoint_path:
        session_name = args.checkpoint_path
        writer = SummaryWriter(session_name)
        best_mean_ap = 0
        # resume if the folder already exists
        if os.path.isdir(args.checkpoint_path):
            architectures.update_incidents_model_with_checkpoint(
                [trunk_model, incident_layer, place_layer], args)
        # otherwise create the folder
        else:
            print("creating new folder with name {}".format(session_name))
    else:
        # in this case, create a new folder with a timestamp
        session_name = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
        print("creating new folder with name {}".format(session_name))
        best_mean_ap = 0
        writer = SummaryWriter(session_name)

    # define the optimizer
    # https://pytorch.org/docs/stable/optim.html#per-parameter-options
    optimizer = torch.optim.Adam(
        [
            {'params': trunk_model.parameters()},
            {'params': incident_layer.parameters()},
            {'params': place_layer.parameters()}
        ],
        lr=args.lr)

    all_models = (trunk_model, incident_layer, place_layer)

    if args.mode == "test":
        print("\n\nRunning in test mode\n\n")
        print("loading test_loader")
        test_loader = get_dataset(args, is_train=False, is_test=True)
        metric = validate(args, test_loader, all_models, epoch=-1, writer=None)
        print("metric on test set: {}".format(metric))
        return
    elif args.mode == "val":
        print("\n\nRunning in val mode\n\n")
        print("loading val_loader")
        val_loader = get_dataset(args, is_train=False)  # TODO: don't shuffle
        metric = validate(args, val_loader, all_models, epoch=-1, writer=None)
        print("metric on val set: {}".format(metric))
        return

    # load train loader in this case
    print("loading train_loader")
    train_loader = get_dataset(args, is_train=True)
    print("loading val_loader")
    val_loader = get_dataset(args, is_train=False)  # TODO: don't shuffle

    for epoch in range(args.start_epoch, args.epochs):

        # train for an epoch
        train(args, train_loader, all_models, optimizer, epoch)

        # evaluate on validation set
        mean_ap = validate(args, val_loader, all_models, epoch=epoch, writer=writer)

        # remember best prec@1 and save checkpoint
        is_best = mean_ap > best_mean_ap
        best_mean_ap = max(mean_ap, best_mean_ap)
        prefix2model = {"trunk": trunk_model,
                        "incident": incident_layer,
                        "place": place_layer}
        # TODO: maybe save at interval, regardless of validation accuracy
        for prefix in prefix2model:
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': prefix2model[prefix].state_dict(),
                'best_mean_ap': best_mean_ap,
            }
            # TODO: need to specify the full path here! and create a folder if needed!
            session_name = args.checkpoint_path
            save_checkpoint(state,
                            is_best,
                            session_name,
                            filename=prefix)


if __name__ == "__main__":
    main()
