import os
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd

file_id_to_filename = {
    # 
    "1HCkYHzXV-nGjRy3iehrr6y90COgG67Jh": "resnet18_places365.pth.tar",
    "1NZHVWK9T-n6mjpAIjrIRtILRV2KcJAlP": "resnet50_places365.pth.tar",
    #
    "1rkCIdcA2YDGMxEoVDk79cp27HakwQLGd": "eccv_baseline_model_trunk.pth.tar",
    "1jj1DJsZgPqS4u4UYOJa_xqq_1ywlinnl": "eccv_baseline_model_incident.pth.tar",
    "1Tovfkt5UP3Nf5moT2QcVqRkB7Nz_D5nh": "eccv_baseline_model_place.pth.tar",
    #
    "1XbPMbKOBkj6EXjMZa4ybOezI6NMZwLOK": "eccv_final_model_trunk.pth.tar",
    "1D3Nh-CqK0jTXu3fgcTwkLSIlQzgApFzu": "eccv_final_model_incident.pth.tar",
    "1DbWnki2352JueF_DrnVmSVvo6bAPlf5W": "eccv_final_model_place.pth.tar",
}

for file_id, filename in tqdm(file_id_to_filename.items()):
    gdd.download_file_from_google_drive(
        file_id=file_id,
        dest_path=os.path.join("pretrained_weights", filename),
    )