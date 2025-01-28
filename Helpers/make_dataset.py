import os
import sys
import shutil
from glob import glob
import numpy as np
import requests
import tarfile
import urllib.request

current_dir = os.path.dirname(os.path.abspath(__file__))
main_path = os.path.join(current_dir, "../")     
sys.path.append(main_path)

DATASET_NAME = "imagewoof"
ROOT_DIR = os.path.join(main_path, "datasets")
DATA_DIR = os.path.join(ROOT_DIR, DATASET_NAME)
RAW_DATA_DIR = os.path.join(ROOT_DIR, f"{DATASET_NAME}2")
SPLITS = ["train", "test"]

def retrieve_data(DATASET_NAME, ROOT_DIR, url):
    output_file = os.path.join(ROOT_DIR, f"{DATASET_NAME}2.tgz")
    RAW_DATA_DIR =  os.path.join(ROOT_DIR, f"{DATASET_NAME}2")
    if os.path.exists(os.path.join(ROOT_DIR, DATASET_NAME)):
        print(f"Dataset already present at : {ROOT_DIR}")
    else:
        if not os.path.exists(output_file):
            print("Downloading dataset ...")
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(output_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)
                print(f"Downloaded {output_file}")
            else:
                print(f"Failed to download. Status code: {response.status_code}")
        if not os.path.exists(RAW_DATA_DIR):
            with tarfile.open(output_file, "r:gz") as tar:
                tar.extractall(path=ROOT_DIR)
            print(f"Extracted files to {ROOT_DIR}")
        else:
            print(f"Extracted file already present at {ROOT_DIR}")


if DATASET_NAME=="imagenette":
    raw_labels = [
        "n01440764", "n02102040", "n02979186", "n03000684", "n03028079", 
        "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"
    ]
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
elif DATASET_NAME=="imagewoof":
    raw_labels = [
        "n02086240", "n02087394", "n02088364", "n02089973", "n02093754", 
        "n02096294", "n02099601",  "n02105641", "n02111889", "n02115641"
    ]
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz"

retrieve_data(DATASET_NAME, ROOT_DIR, url)

train_folders = sorted(glob(os.path.join(RAW_DATA_DIR, "train", "*")))

class_maps = {}
for i in range(len(raw_labels)):
    class_maps[raw_labels[i]] = str(i)

class_names = list(class_maps.keys())
class_indices = list(class_maps.values())
n_classes=len(class_indices)

assert len(train_folders)==len(class_maps)

for sp in SPLITS:
    for cls in class_indices:
        os.makedirs(os.path.join(DATA_DIR, sp, cls), exist_ok=True)

for i, cls_index in enumerate(class_indices):
    class_idx = class_indices[i]
    image_paths = np.array(glob(os.path.join(f"{train_folders[int(class_idx)]}","*.JPEG")))
    class_idx = class_indices[i]
    print(f'{class_idx}: {len(image_paths)}')
    np.random.shuffle(image_paths)
    ds_split = np.split(
        image_paths, 
        indices_or_sections=[int(.9*len(image_paths)), int(1*len(image_paths))]
    )
    dataset_data = zip(SPLITS, ds_split)
    for sp, images in dataset_data:
        for img_path in images:
            shutil.copy(img_path, os.path.join(DATA_DIR, sp, class_idx))

shutil.rmtree(os.path.join(RAW_DATA_DIR))
os.remove(os.path.join(ROOT_DIR, f"{DATASET_NAME}2.tgz"))
