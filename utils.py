"""utils.py

Helping code for rest of repo.
"""

import pickle
import os
import shutil
import torch
import json


def get_place_to_index_mapping():
    place_to_index_mapping = {}
    file1 = open("categories/places.txt", "r")
    lines = [line.rstrip() for line in file1.readlines()]
    for idx, place in enumerate(lines):
        place_to_index_mapping[place] = idx
    file1.close()
    return place_to_index_mapping


def get_index_to_place_mapping():
    x = get_place_to_index_mapping()
    # https://dev.to/renegadecoder94/how-to-invert-a-dictionary-in-python-2150
    x = dict(map(reversed, x.items()))
    return x


def get_incident_to_index_mapping():
    incident_to_index_mapping = {}
    file1 = open("categories/incidents.txt", "r")
    lines = [line.rstrip() for line in file1.readlines()]
    for idx, incident in enumerate(lines):
        incident_to_index_mapping[incident] = idx
    file1.close()
    return incident_to_index_mapping


def get_index_to_incident_mapping():
    x = get_incident_to_index_mapping()
    # https://dev.to/renegadecoder94/how-to-invert-a-dictionary-in-python-2150
    x = dict(map(reversed, x.items()))
    return x


def get_loaded_pickle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state,
                    is_best,
                    session_name,
                    filename='checkpoint'):
    path = os.path.join(session_name, "_{}.pth.tar".format(filename))
    best_path = os.path.join(session_name, "_{}_best.pth.tar".format(filename))
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, best_path)


def get_loaded_json_file(path):
    with open(path, "r") as fp:
        return json.load(fp)
