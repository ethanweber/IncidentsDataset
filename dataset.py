"""dataset.py

Code to use the datasets.
"""

import random
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as transforms
from collections import defaultdict

from utils import (
    get_place_to_index_mapping,
    get_incident_to_index_mapping,
    get_loaded_json_file
)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def image_loader(filename):
    with open(filename, 'rb') as f:
        image = Image.open(f).convert('RGB')
    return image


def get_vectors(data, to_index_mapping, vector_len):
    """Get the vector of labels and weight vector for our labeled iamges.

    Args:
        data (dict): 
            {
                "place_or_disater": +1/-1
            }
        to_index_mapping (dict):
            {
                "place_or_disater": index (int)
            }
        vector_len (int): lenth of vector (includes "no place" or "no incident")

    Returns:
        tuple: (vector with +1/0/-1 for labels, weight vector with +1 where we have information)
    """
    vector = np.zeros(vector_len)
    weight_vector = np.zeros(vector_len)
    for key, value in data.items():
        index = to_index_mapping[key]
        if value == 1:
            vector[index] = 1
            weight_vector = np.ones(vector_len)  # assume full information
        elif value == 0:  # TODO: fix this hack for now
            weight_vector[index] = 1
        else:
            raise ValueError("dict should be sparse, with just 1 and 0")
    return vector, weight_vector


def get_split_dictionary(data):
    splits = []
    if len(data["incidents"]) == 0:
        for key, value in data["places"].items():
            splits.append({
                "incidents": {},
                "places": {key: value}
            })
    elif len(data["places"]) == 0:
        for key, value in data["incidents"].items():
            splits.append({
                "incidents": {key: value},
                "places": {}
            })
    else:
        for d, dv in data["incidents"].items():
            for p, pv in data["places"].items():
                splits.append({
                    "incidents": {d: dv},
                    "places": {p: pv}
                })
    return splits


class IncidentDataset(Dataset):
    """A Pytorch dataset for classification of incidents images with incident and place.

    Args:
        incidents_images (dict): Images that are part of our dataset.
        place_to_index_mapping (dict):
        incident_to_index_mapping (dict):

    Attributes:
        place_names (list): List of the place names.
        place_name_to_idx (dict): Dict with items (place_name, index).
        incident_name (list): List of the incident names.
        incident_name_to_idx (dict): Dict with items (incident_name, index).
    """

    def __init__(
            self,
            images_path,
            incidents_images,
            place_to_index_mapping,
            incident_to_index_mapping,
            transform=None,
            use_all=False,
            pos_only=False,
            using_softmax=False):

        self.images_path = images_path
        self.use_all = use_all

        self.items = []
        self.all_data = []
        self.no_incident_label_items = []  # items without a incident label

        self.no_incidents = defaultdict(list)

        print("adding incident images")
        for filename, original_data in tqdm(incidents_images.items()):

            splits = get_split_dictionary(original_data)
            for data in splits:

                assert len(data["incidents"]) <= 1 and len(data["places"]) <= 1

                if using_softmax:
                    # the +1 to len accounts for "no place" and "no incident"
                    place_vector, place_weight_vector = get_vectors(
                        data["places"], place_to_index_mapping, len(place_to_index_mapping) + 1)
                    incident_vector, incident_weight_vector = get_vectors(
                        data["incidents"], incident_to_index_mapping, len(incident_to_index_mapping) + 1)
                else:
                    place_vector, place_weight_vector = get_vectors(
                        data["places"], place_to_index_mapping, len(place_to_index_mapping))
                    incident_vector, incident_weight_vector = get_vectors(
                        data["incidents"], incident_to_index_mapping, len(incident_to_index_mapping))

                # TODO: need to add "no incident" to some...
                # TODO: somehow fix this hack
                # means its part of the places dataset, so no incident
                if len(data["incidents"]) == 0 and using_softmax == True:
                    incident_vector[-1] = 1  # "no incident" is +1
                    incident_weight_vector = np.ones(
                        len(incident_weight_vector))
                elif len(data["incidents"]) == 0:
                    incident_vector = np.zeros(len(incident_weight_vector))
                    incident_weight_vector = np.ones(
                        len(incident_weight_vector))

                # choose which set to put them into
                has_incident = False
                for label in data["incidents"].values():
                    if label == 1:
                        has_incident = True
                        break

                item = (filename, place_vector, incident_vector,
                        place_weight_vector, incident_weight_vector)

                if pos_only:
                    if has_incident:
                        self.all_data.append(item)
                    else:
                        pass
                else:
                    self.all_data.append(item)

        print("number items: {}".format(len(self.all_data)))
        self.image_loader = image_loader
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: incident_label_item (list), no_incident_label_item (list)
        """

        my_item = list(self.all_data[index])
        image_name = my_item[0]
        img = self.image_loader(os.path.join(self.images_path, image_name))
        if self.transform is not None:
            img = self.transform(img)
        my_item[0] = img
        return my_item


# TODO: change to dataloader, not dataset
def get_dataset(args,
                is_train=True,
                is_test=False):
    """

    :param args:
    :param is_train:
    :param is_test:
    :return:
    """
    # """Returns the dataset for training or testing.
    #
    # Args:
    #     args:
    #
    # Returns:
    #     DataLoader:
    # """

    # main dataset (incidents images)
    if is_train:
        incidents_images = get_loaded_json_file(args.dataset_train)
    else:
        if is_test == False:  # validation images
            incidents_images = get_loaded_json_file(args.dataset_val)
        else:  # test images
            incidents_images = get_loaded_json_file(args.dataset_test)

    place_to_index_mapping = get_place_to_index_mapping()
    incident_to_index_mapping = get_incident_to_index_mapping()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        pos_only = args.dataset == "pos_only"
        shuffle = True
        use_all = False
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        pos_only = False
        shuffle = False
        use_all = True
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    using_softmax = args.activation == "softmax"

    dataset = IncidentDataset(
        args.images_path,
        incidents_images,
        place_to_index_mapping,
        incident_to_index_mapping,
        transform=transform,
        use_all=use_all,
        pos_only=pos_only,
        using_softmax=using_softmax
    )

    # TODO: avoid the double shuffling affect that currently exists
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True
    )

    return loader
