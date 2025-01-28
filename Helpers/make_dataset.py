import os
import sys
sys.path.append("../")

DATASET_NAME = "imagenette"
ROOT_DIR = "../dataset"
DATA_DIR = os.path.join(ROOT_DIR, DATASET_NAME)
RAW_DATA_DIR = os.path.join(ROOT_DIR, f"{DATASET_NAME}2")
SPLITS = ["train", "test"]


def retrieve_data(DATASET_NAME, ROOT_DIR, url):


if DATASET_NAME=="imagenette":
    raw_labels = [
        "n01440764", "n02102040", "n02979186", "n03000684", "n03028079", 
        "n03394916", "n03417042", "n03425413", "n03445777", "n03888257"
    ]
elif DATASET_NAME=="imagewoof":
    raw_labels = [
        "n02086240", "n02087394", "n02088364", "n02089973", "n02093754", 
        "n02096294", "n02099601",  "n02105641", "n02111889", "n02115641"
    ]


train_folders = sorted(glob(os.path.join(RAW_DATA_DIR, "*")))

class_maps = {}
for i in range(raw_labels):
    class_maps[raw_labels[i]] = str(i)

class_names = list(class_maps.keys())
class_indices = list(class_maps.values())
n_classes=len(class_indices)

assert len(train_folders)==len(class_maps)

for sp in SPLITS:
  for cls in class_indexes:
    os.makedirs(os.path.join(DATA_DIR, sp, cls), exist_ok=True)

for i, cls_index in enumerate(class_indices):
  image_paths = np.array(glob(f'{train_folders[class_indices [cls_index]]}/*.JPEG'))
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

os.remove(os.path.join(RAW_DATA_DIR))
os.remove(os.path.join(ROOT_DIR, f"{DATASET_NAME}2.tgz"))
