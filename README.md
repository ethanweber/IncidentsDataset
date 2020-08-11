# Incidents Dataset

See the following pages for more details:
 - Project page: [IncidentsDataset.csail.mit.edu](http://incidentsdataset.csail.mit.edu/).
 - ECCV 2020 Paper "Detecting natural disasters and damage in the wild" [here]().

# Obtain the data
> Note that the data is not accessible yet! It's undergoing sensitive content and bias analysis. If you have any questions or would like data access sooner than public release, please contact incidentsdataset@googlegroups.com. The pretrained model weights, however, are available and should work as expected.

Download the data at [here](https://drive.google.com/drive/folders/1kPn0u6jghhaAv_1Nj7tMcPkSLkokNzTk?usp=sharing). The data structure is in JSON with URLs and labels. We provide code to download the images from URLs. The files are in the following form:

```
# single-label multi-class (ECCV 2020 version):
eccv_train.json
eccv_val.json

# multi-label multi-class (latest version):
multi_label_train.json
multi_label_val.json
```

1. Download chosen JSON files and move to the [data](data/) folder.

2. Look at [VisualizeDataset.ipynb](VisualizeDataset.ipynb) to see the composition of the dataset files.

3. Download the images specified in the JSON files.

    ```
    cd data/
    # run this and follow instructions
    python run_download_images.py --help
    ```
   
4. Take note of image download location. This is param `--images_path` in [parser.py](/parser).

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
   
2. Run inference with the model with [RunModel.ipynb](RunModel.ipynb).

# TODO(ethan): allow option to run this on the validation set, since we won't release the test set
3. Compute mAP and report numbers.
    ```
    # test the model on the validation set
    python run_model.py \
        --config=configs/eccv_final_model \
        --mode=val \
        --checkpoint_path=pretrained_weights \
        --images_path=/path/to/downloaded/images/folder/
    ```

4. Train a model.
    ```
    # train the model
    python run_model.py \
        --config=configs/eccv_final_model \
        --mode=train \
        --checkpoint_path=runs/eccv_final_model
   
    # visualize tensorboard
    tensorboard --samples_per_plugin scalars=100,images=10 --port 8880 --bind_all --logdir runs/eccv_final_model
    ```