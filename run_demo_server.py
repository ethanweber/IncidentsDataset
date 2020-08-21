from PIL import Image
from flask import Flask, jsonify, request
from io import BytesIO
import base64
import re
import torchvision.transforms as transforms

from architectures import (
    FilenameDataset,
    get_incidents_model,
    update_incidents_model_with_checkpoint,
    update_incidents_model_to_eval_mode,
    get_predictions_from_model
)
from parser import get_parser, get_postprocessed_args
from utils import get_index_to_incident_mapping, get_index_to_place_mapping

app = Flask(__name__)
app.jinja_env.filters['zip'] = zip

# model
CONFIG_FILENAME = "configs/eccv_final_model"
CHECKPOINT_PATH_FOLDER = "pretrained_weights/"

# Load model from checkpoint.
parser = get_parser()
args = parser.parse_args(
    args="--config={} --checkpoint_path={} --mode=test --num_gpus=1".format(CONFIG_FILENAME, CHECKPOINT_PATH_FOLDER))
args = get_postprocessed_args(args)

incidents_model = get_incidents_model(args)
update_incidents_model_with_checkpoint(incidents_model, args)
update_incidents_model_to_eval_mode(incidents_model)

# transform for inference
inference_loader = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def decode_base64(data, altchars=b'+/'):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    data = re.sub(rb'[^a-zA-Z0-9%s]+' % altchars, b'', data)  # normalize
    missing_padding = len(data) % 4
    if missing_padding:
        data += b'=' * (4 - missing_padding)
    return base64.b64decode(data, altchars)


# Endpoint to get the prediction.
@app.route('/prediction', methods=["POST"])
def prediction():
    s = request.form.get('base64').replace(" ", "+")
    imgdata = base64.b64decode(s)
    im = Image.open(BytesIO(imgdata)).convert("RGB")
    image = inference_loader(im)
    batch_input = image[None]  # add batch dim

    image_paths = ["imagefilename"]
    inference_dict = {}
    output = get_predictions_from_model(
        args,
        incidents_model,
        batch_input,
        image_paths,
        get_index_to_incident_mapping(),
        get_index_to_place_mapping(),
        inference_dict,
        topk=5
    )

    result = {}
    result["incidents"] = []
    result["places"] = []
    inf_dict = inference_dict["imagefilename"]
    for dis, prob in zip(inf_dict["incidents"], inf_dict["incident_probs"]):
        result["incidents"].append([dis, round(float(prob), 2)])
    for pl, prob in zip(inf_dict["places"], inf_dict["place_probs"]):
        result["places"].append([pl, round(float(prob), 2)])
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    print("starting main")
    app.run(debug=True, threaded=True, host="0.0.0.0", port=8012)
