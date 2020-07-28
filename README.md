# Incidents Dataset

See the following pages for more details:
 - Project page: [IncidentsDataset.csail.mit.edu](http://incidentsdataset.csail.mit.edu/).
 - ECCV 2020 Paper "Detecting natural disasters and damage in the wild" [here]().

# Obtain the data

Download the data at [here](https://drive.google.com/drive/folders/1kPn0u6jghhaAv_1Nj7tMcPkSLkokNzTk?usp=sharing). The data structure is in JSON with URLs and labels. We provide code to download the images from URLs. The files are in the folliwing form:

```
# single-label multi-class (ECCV 2020 version):
eccv_train.json
eccv_val.json
eccv_test.json

# multi-label multi-class (latest version):
multi_label_train.json
multi_label_val.json
multi_label_test.json
```

1. Download chosen JSON files and move to the [data](data/) folder.

2. Look at [VisualizeDataset.ipynb](VisualizeDataset.ipynb) to see the composition of the dataset files.

3. Download the images specified in the JSON files.

    ```
    cd data/
    # run this and follow instructions
    python run_download_images.py --help
    ```

# Setup environment



# Using the Incident Model

1. Download pretrained weights [here](https://drive.google.com/drive/folders/1k2nggK3LqyBE5huGpL3E-JXoEv7o6qRq?usp=sharing). Place desired files in the [pretrained_weights](pretrained_weights/) folder. Note that these take the following structure:

    ```
    # pretrained weights with Places 365
    resnet18_places365.pth.tar
    resnet50_places365.pth.tar
    
    # ECCV baseline model weights
    eccv_baseline_model_trunk.pth.tar
    eccv_baseline_model_incident.pth.tar
    eccv_baseline_model_place.pth.tar
    
    # ECCV final model weights
    eccv_final_model_trunk.pth.tar
    eccv_final_model_incident.pth.tar
    eccv_final_model_place.pth.tar
    ```

```
# train the model
python run_model.py --config=configs/eccv_final_model --mode=train

# test the model
# TODO: set the checkpoint
python run_model.py --config=configs/eccv_final_model --mode=test
```