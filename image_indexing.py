import json
from tensorflow.keras.utils import get_file
# Download ImageNet class index mapping (JSON file)
json_path = get_file("imagenet_class_index.json",
                     "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")
# Load JSON file
with open(json_path, "r") as f:
    class_index = json.load(f)
# Print all 1,000 classes
for idx, (imagenet_id, label) in class_index.items():
    print(f"Index {idx}: {label}")

import requests
# Download ImageNet class labels
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = requests.get(url).text.split("\n")
# Print all classes
for idx, label in enumerate(imagenet_classes):
    print(f"Index {idx}: {label}")