"""loss.py

Get the loss for the Incident Model.
"""
import torch
import torch.nn as nn


def get_loss(args,
             incident_output,
             incident_target,
             incident_weight,
             place_output,
             place_target,
             place_weight):
    """
    
    Args:
        args: 
        incident_output: tensor of logits 
        incident_target: tensor with 1s and 0s representing the GT label (default 0)
        incident_weight: tensor with 1s where we have information
        place_output: 
        place_target: 
        place_weight: 
        is_train: 

    Returns:
        torch.Tensor: a scalar for the loss
    """

    # pass through desired activation
    if args.activation == "softmax":
        m = nn.Softmax(dim=1)
    elif args.activation == "sigmoid":
        m = nn.Sigmoid()
    incident_output = m(incident_output) # [B, 43]
    place_output = m(place_output) # [B, 49]

    criterion = nn.BCELoss(reduction='none')
    incident_loss = torch.sum(
        criterion(
            incident_output,
            incident_target.type(torch.FloatTensor).cuda(non_blocking=True)
        ) * incident_weight, dim=1) # to shape [B]
    # amplify the loss by the number of positive labels
    # if no positive labels, then multiply by ones
    # multiplier = torch.clamp(torch.sum(incident_target, dim=1), min=1)
    # incident_loss = (incident_loss * multiplier).mean()
    incident_loss = incident_loss.mean()

    place_loss = torch.sum(
        criterion(
            place_output,
            place_target.type(torch.FloatTensor).cuda(non_blocking=True)
        ) * place_weight, dim=1)
    # multiplier = torch.clamp(torch.sum(place_target, dim=1), min=1)
    # place_loss = (place_loss * multiplier).mean()
    place_loss = place_loss.mean()
    loss = incident_loss + place_loss
    return loss, incident_output, place_output
