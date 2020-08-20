# Incidents Dataset

See the following pages for more details:
 - Project page: [IncidentsDataset.csail.mit.edu](http://incidentsdataset.csail.mit.edu/).
 - ECCV 2020 Paper "Detecting natural disasters, damage, and incidents in the wild" [here](http://incidentsdataset.csail.mit.edu/IncidentsDatasetPaper.pdf).

# Obtain the data
> Note that the data is not accessible yet. If you have any questions or would like data access sooner than public release, please contact incidentsdataset@googlegroups.com.

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

```
git clone https://github.com/ethanweber/IncidentsDataset
cd IncidentsDataset

conda create -n incidents python=3.8.2
conda activate incidents
pip install -r requirements.txt
```

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
   
# Citation

If you find this work helpful for your research, please consider citing our paper:

```
@InProceedings{weber2020cvpr,
  title={Detecting natural disasters, damage, and incidents in the wild},
  author={Weber, Ethan and Marzo, Nuria and Papadopoulos, Dim P. and Biswas, Aritro and Lapedriza, Agata and Ofli, Ferda and Imran, Muhammad and Torralba, Antonio},
  booktitle={The European Conference on Computer Vision (ECCV)},
  month = {August},
  year={2020}
}
```

# License

This work is licensed with the MIT License. See [LICENSE](LICENSE) for details.

# Acknowledgements

This work is supported by the CSAIL-QCRI collaboration project and RTI2018-095232-B-C22 grant from the Spanish Ministry of Science, Innovation and Universities.