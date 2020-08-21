"""models.py
"""
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from skimage import io

import numpy as np
import os

from PIL import Image

# same loader used during training
inference_loader = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class FilenameDataset(data.Dataset):
    """
    Data loader for filenames and their corresponding labels.
    """

    def __init__(self, image_filenames, targets):
        """
        Args:
            image_filenames (list): List of image filenames
            targets (list): List of integers that correspond to target class indices
        """
        assert (len(image_filenames) == len(targets))
        self.image_filenames = image_filenames
        self.targets = targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the target class index
        """
        image_filename = self.image_filenames[index]
        # if not os.path.isfile(image_filename):
        #     os.system("ln -s {} {}".format(image_filename.replace("/data/vision/torralba/humanitarian/datasets/images_raw/",
        #                                                           "/data/vision/torralba/humanitarian/dimitris/getGoogleImages2/finalImages/"), image_filename))
        if not os.path.isfile(image_filename):
            raise ValueError("{} is not a file".format(image_filename))
        try:
            with open(image_filename, 'rb') as f:
                image = Image.open(f).convert('RGB')
                image = inference_loader(image)
        except:
            print(image_filename)
            image = Image.new('RGB', (300, 300), 'white')
            image = inference_loader(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.image_filenames)


def get_trunk_model(args):
    if args.pretrained_with_places:
        print("loading places weights for pretraining")
        model = models.__dict__[args.arch](num_classes=365)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.arch == "resnet18":
            model_file = os.path.join(dir_path, "pretrained_weights/resnet18_places365.pth.tar")
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = {str.replace(k, 'module.', ''): v for k,
                                                               v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.fc = nn.Linear(512, 1024)
            model = nn.Sequential(model, nn.ReLU())
        elif args.arch == "resnet50":
            model_file = os.path.join(dir_path, "pretrained_weights/resnet50_places365.pth.tar")
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = {str.replace(k, 'module.', ''): v for k,
                                                               v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.fc = nn.Linear(2048, 1024)
            model = nn.Sequential(model, nn.ReLU())
        return model
    else:
        print("loading imagenet weights for pretraining")
        # otherwise load with imagenet weights
        if args.arch == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(512, 1024)
            model = nn.Sequential(model, nn.ReLU())
        elif args.arch == "resnet50":
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(2048, 1024)
            model = nn.Sequential(model, nn.ReLU())
        return model


def get_incident_layer(args):
    if args.activation == "softmax":
        return nn.Linear(args.fc_dim, args.num_incidents + 1)
    elif args.activation == "sigmoid":
        return nn.Linear(args.fc_dim, args.num_incidents)


def get_place_layer(args):
    if args.activation == "softmax":
        return nn.Linear(args.fc_dim, args.num_places + 1)
    elif args.activation == "sigmoid":
        return nn.Linear(args.fc_dim, args.num_places)


def get_incidents_model(args):
    """
    Returns [trunk_model, incident_layer, place_layer]
    """
    # the shared feature trunk model
    trunk_model = get_trunk_model(args)
    # the incident model
    incident_layer = get_incident_layer(args)
    # the place model
    place_layer = get_place_layer(args)

    print("Let's use", args.num_gpus, "GPUs!")
    trunk_model = torch.nn.DataParallel(trunk_model, device_ids=range(args.num_gpus))
    incident_layer = torch.nn.DataParallel(incident_layer, device_ids=range(args.num_gpus))
    place_layer = torch.nn.DataParallel(place_layer, device_ids=range(args.num_gpus))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk_model.to(device)
    incident_layer.to(device)
    place_layer.to(device)
    return [trunk_model, incident_layer, place_layer]


def update_incidents_model_with_checkpoint(incidents_model, args):
    """
    Update incidents model with checkpoints (in args.checkpoint_path)
    """

    trunk_model, incident_layer, place_layer = incidents_model

    # optionally resume from a checkpoint
    # TODO: bring in the original pretrained weights maybe?
    # TODO: remove the args.trunk_resume, etc.
    # TODO: remove path prefix

    config_name = os.path.basename(args.config)
    print(config_name)

    trunk_resume = os.path.join(
        args.checkpoint_path, "{}_trunk.pth.tar".format(config_name))
    place_resume = os.path.join(
        args.checkpoint_path, "{}_place.pth.tar".format(config_name))
    incident_resume = os.path.join(
        args.checkpoint_path, "{}_incident.pth.tar".format(config_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for (path, net) in [(trunk_resume, trunk_model), (place_resume, place_layer), (incident_resume, incident_layer)]:
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}' (epoch {}).".format(path, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'.".format(path))


def update_incidents_model_to_eval_mode(incidents_model):
    print("Switching to eval mode.")
    for m in incidents_model:
        # switch to evaluation mode
        m.eval()


def get_predictions_from_model(args,
                               incidents_model,
                               batch_input,
                               image_paths,
                               index_to_incident_mapping,
                               index_to_place_mapping,
                               inference_dict, topk=1):
    """
    Input:
    {
        "image_paths" = [list of image paths],
    }
    Returns {
        "incidents": [], # list of topk elements
        "places": [] # list of topk elements
    }
    """
    trunk_model, incident_layer, place_layer = incidents_model

    # compute output with models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = batch_input.to(device)
    output = trunk_model(input)
    incident_output = incident_layer(output)
    place_output = place_layer(output)

    if args.activation == "softmax":
        incident_output = F.softmax(incident_output, dim=1)
        place_output = F.softmax(place_output, dim=1)
    elif args.activation == "sigmoid":
        m = nn.Sigmoid()
        incident_output = m(incident_output)
        place_output = m(place_output)

    incident_probs, incident_idx = incident_output.sort(1, True)
    place_probs, place_idx = place_output.sort(1, True)

    # batch_input[0] is the batch dimension (the # in the batch)
    for batch_idx in range(len(batch_input.numpy())):
        incidents = []
        for idx in incident_idx[batch_idx].cpu().numpy()[:topk]:
            if idx < len(index_to_incident_mapping):
                incidents.append(
                    index_to_incident_mapping[idx]
                )
            else:
                incidents.append("no incident")

        places = []
        for idx in place_idx[batch_idx].cpu().numpy()[:topk]:
            if idx < len(index_to_place_mapping):
                places.append(
                    index_to_place_mapping[idx]
                )
            else:
                places.append("no place")

        output = {
            "incidents": incidents,
            "places": places,
            "incident_probs": incident_probs[batch_idx].cpu().detach().numpy()[:topk],
            "place_probs": place_probs[batch_idx].cpu().detach().numpy()[:topk]
        }
        image_path = image_paths[batch_idx]
        inference_dict[image_path] = output

    # TODO: maybe return the output here
    return None


def get_predictions_from_model_all(args, incidents_model, batch_input, image_paths, index_to_incident_mapping,
                                   index_to_place_mapping, inference_dict, softmax=True):
    """
    Input:
    {
        "image_paths" = [list of image paths],
    }
    Returns {
        "incidents": [], # list of topk elements
        "places": [] # list of topk elements
    }
    """
    trunk_model, incident_layer, place_layer = incidents_model

    # compute output with models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = batch_input.to(device)
    output = trunk_model(input)
    incident_output = incident_layer(output)
    place_output = place_layer(output)

    if softmax:
        incident_output = F.softmax(incident_output, dim=1)
        place_output = F.softmax(place_output, dim=1)
    else:
        m = nn.Sigmoid()
        incident_output = m(incident_output)
        place_output = m(place_output)

    incident_probs, incident_idx = incident_output.sort(1, True)
    place_probs, place_idx = place_output.sort(1, True)

    # batch_input[0] is the batch dimension (the # in the batch)
    for batch_idx in range(len(batch_input.numpy())):
        incidents = []
        for idx in incident_idx[batch_idx].cpu().numpy():
            if idx < len(index_to_incident_mapping):
                incidents.append(
                    index_to_incident_mapping[idx]
                )
            else:
                incidents.append("no incident")

        places = []
        for idx in place_idx[batch_idx].cpu().numpy():
            if idx < len(index_to_place_mapping):
                places.append(
                    index_to_place_mapping[idx]
                )
            else:
                places.append("no place")

        output = {
            "incidents": incidents,
            "places": places,
            "incident_probs": incident_probs[batch_idx].cpu().detach().numpy(),
            "place_probs": place_probs[batch_idx].cpu().detach().numpy()
        }
        image_path = image_paths[batch_idx]
        inference_dict[image_path] = output

    # TODO: maybe return the output here
    return None


def get_features_from_model(incidents_model, batch_input, image_paths, inference_dict):
    """
    Input:
    {
        "image_paths" = [list of image paths],
    }
    Returns trunk_model output.
    """
    trunk_model, incident_layer, place_layer = incidents_model

    # compute output with models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = batch_input.to(device)
    output = trunk_model(input)

    # batch_input[0] is the batch dimension (the # in the batch)
    for batch_idx in range(len(batch_input.numpy())):
        out = output[batch_idx].cpu().detach().numpy()
        # print("here")
        # print(out)
        # print(out.shape)
        # print(type(out))
        image_path = image_paths[batch_idx]
        inference_dict[image_path] = out

    # TODO: maybe return the output here
    return None
