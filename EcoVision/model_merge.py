# import os
# import zipfile
# import shutil
# import yaml

# # ======================================================
# # ZIP FILES
# # ======================================================

# ZIP_1 = "FYP Dataset 1.v1i.yolov11.zip"
# ZIP_2 = "recycle.v1i.yolov11.zip"

# EXTRACT_1 = "FYP Dataset 1.v1i.yolov11"
# EXTRACT_2 = "recycle.v1i.yolov11"

# MERGED = "merged_dataset"

# # ======================================================
# # FINAL CLASS ORDER
# # ======================================================

# FINAL_CLASSES = [
#     "can",
#     "cardboard",
#     "glass",
#     "plastic",
#     "metal",
#     "null",
#     "paper",
#     "trash"
# ]

# # ======================================================
# # EXTRACT ZIP FILES
# # ======================================================

# def extract_zip(zip_path, extract_to):
#     if os.path.exists(extract_to):
#         print(f"Already extracted: {extract_to}")
#         return

#     with zipfile.ZipFile(zip_path, "r") as zip_ref:
#         zip_ref.extractall(extract_to)

#     print(f"Extracted: {zip_path}")


# extract_zip(ZIP_1, EXTRACT_1)
# extract_zip(ZIP_2, EXTRACT_2)

# # ======================================================
# # FIND data.yaml
# # ======================================================

# def find_yaml(folder):
#     for root, dirs, files in os.walk(folder):
#         if "data.yaml" in files:
#             return os.path.join(root, "data.yaml")
#     return None


# yaml1 = find_yaml(EXTRACT_1)
# yaml2 = find_yaml(EXTRACT_2)

# if yaml1 is None or yaml2 is None:
#     raise FileNotFoundError("Could not find data.yaml")

# print("Dataset 1 YAML:", yaml1)
# print("Dataset 2 YAML:", yaml2)

# # ======================================================
# # READ CLASS NAMES
# # ======================================================

# with open(yaml1, "r") as f:
#     data1 = yaml.safe_load(f)

# with open(yaml2, "r") as f:
#     data2 = yaml.safe_load(f)

# print("\nDataset 1 classes:")
# print(data1["names"])

# print("\nDataset 2 classes:")
# print(data2["names"])

# # ======================================================
# # CREATE CLASS MAPPING
# # ======================================================

# def build_mapping(dataset_names):
#     mapping = {}

#     if isinstance(dataset_names, dict):
#         iterator = dataset_names.items()
#     else:
#         iterator = enumerate(dataset_names)

#     for old_id, class_name in iterator:
#         old_id = int(old_id)

#         if class_name not in FINAL_CLASSES:
#             raise ValueError(f"Unknown class: {class_name}")

#         new_id = FINAL_CLASSES.index(class_name)

#         mapping[old_id] = new_id

#     return mapping


# mapping1 = build_mapping(data1["names"])
# mapping2 = build_mapping(data2["names"])

# print("\nDataset 1 mapping:")
# print(mapping1)

# print("\nDataset 2 mapping:")
# print(mapping2)

# # ======================================================
# # RESET / CREATE OUTPUT FOLDERS
# # ======================================================

# if os.path.exists(MERGED):
#     shutil.rmtree(MERGED)

# splits = ["train", "valid", "test"]

# for split in splits:
#     os.makedirs(
#         os.path.join(MERGED, split, "images"),
#         exist_ok=True
#     )

#     os.makedirs(
#         os.path.join(MERGED, split, "labels"),
#         exist_ok=True
#     )

# # ======================================================
# # REMAP LABEL FILE
# # ======================================================

# def remap_label_file(src_file, dst_file, mapping):
#     new_lines = []

#     with open(src_file, "r") as f:
#         lines = f.readlines()

#     for line in lines:
#         parts = line.strip().split()

#         if len(parts) < 5:
#             continue

#         old_class = int(parts[0])

#         if old_class not in mapping:
#             raise ValueError(
#                 f"Class ID {old_class} not found in mapping for {src_file}"
#             )

#         new_class = mapping[old_class]

#         parts[0] = str(new_class)

#         new_lines.append(" ".join(parts))

#     with open(dst_file, "w") as f:
#         f.write("\n".join(new_lines))

# # ======================================================
# # COPY DATASET
# # ======================================================

# def merge_dataset(dataset_path, mapping, prefix):
#     for split in splits:
#         img_src = os.path.join(dataset_path, split, "images")
#         lbl_src = os.path.join(dataset_path, split, "labels")

#         img_dst = os.path.join(MERGED, split, "images")
#         lbl_dst = os.path.join(MERGED, split, "labels")

#         if os.path.exists(img_src):
#             for file in os.listdir(img_src):
#                 src = os.path.join(img_src, file)

#                 dst = os.path.join(
#                     img_dst,
#                     f"{prefix}_{file}"
#                 )

#                 shutil.copy2(src, dst)

#         if os.path.exists(lbl_src):
#             for file in os.listdir(lbl_src):
#                 src = os.path.join(lbl_src, file)

#                 dst = os.path.join(
#                     lbl_dst,
#                     f"{prefix}_{file}"
#                 )

#                 remap_label_file(
#                     src,
#                     dst,
#                     mapping
#                 )


# merge_dataset(EXTRACT_1, mapping1, "d1")
# merge_dataset(EXTRACT_2, mapping2, "d2")

# print("\nDatasets merged")

# # ======================================================
# # CREATE FINAL data.yaml
# # ======================================================

# yaml_data = {
#     "train": "train/images",
#     "val": "valid/images",
#     "test": "test/images",
#     "nc": len(FINAL_CLASSES),
#     "names": FINAL_CLASSES
# }

# yaml_path = os.path.join(MERGED, "data.yaml")

# with open(yaml_path, "w") as f:
#     yaml.dump(yaml_data, f, sort_keys=False)

# print("\nFinal data.yaml created")

# print("\nMerged dataset ready:")
# print(MERGED)

# print("\nFinal classes:")
# for i, name in enumerate(FINAL_CLASSES):
#     print(f"{i}: {name}")



import os
import shutil
import yaml

# ======================================================
# PATHS
# ======================================================

OLD_DATASET = "merged_dataset"
NEW_DATASET = "merged_dataset_cleaned"

OLD_YAML = os.path.join(OLD_DATASET, "data.yaml")

# ======================================================
# FINAL CLASSES
# ======================================================

FINAL_CLASSES = [
    "cardboard",
    "glass",
    "plastic",
    "metal",
    "empty",
    "paper",
    "trash"
]

# Old class names from your current merged dataset
OLD_CLASSES = [
    "can",
    "cardboard",
    "glass",
    "plastic",
    "metal",
    "null",
    "paper",
    "trash"
]

# Class remapping
CLASS_REMAP = {
    "can": "metal",
    "cardboard": "cardboard",
    "glass": "glass",
    "plastic": "plastic",
    "metal": "metal",
    "null": "empty",
    "paper": "paper",
    "trash": "trash"
}

# ======================================================
# RESET OUTPUT FOLDER
# ======================================================

if os.path.exists(NEW_DATASET):
    shutil.rmtree(NEW_DATASET)

splits = ["train", "valid", "test"]

for split in splits:
    os.makedirs(os.path.join(NEW_DATASET, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(NEW_DATASET, split, "labels"), exist_ok=True)

# ======================================================
# REMAP LABEL FILE
# ======================================================

def remap_label_file(src_label, dst_label):
    new_lines = []

    with open(src_label, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        old_id = int(parts[0])
        old_class = OLD_CLASSES[old_id]

        new_class = CLASS_REMAP[old_class]
        new_id = FINAL_CLASSES.index(new_class)

        parts[0] = str(new_id)
        new_lines.append(" ".join(parts))

    with open(dst_label, "w") as f:
        f.write("\n".join(new_lines))

# ======================================================
# COPY + REMAP DATASET
# ======================================================

for split in splits:
    old_img_dir = os.path.join(OLD_DATASET, split, "images")
    old_lbl_dir = os.path.join(OLD_DATASET, split, "labels")

    new_img_dir = os.path.join(NEW_DATASET, split, "images")
    new_lbl_dir = os.path.join(NEW_DATASET, split, "labels")

    if os.path.exists(old_img_dir):
        for file in os.listdir(old_img_dir):
            shutil.copy2(
                os.path.join(old_img_dir, file),
                os.path.join(new_img_dir, file)
            )

    if os.path.exists(old_lbl_dir):
        for file in os.listdir(old_lbl_dir):
            remap_label_file(
                os.path.join(old_lbl_dir, file),
                os.path.join(new_lbl_dir, file)
            )

# ======================================================
# CREATE NEW data.yaml
# ======================================================

yaml_data = {
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": len(FINAL_CLASSES),
    "names": FINAL_CLASSES
}

with open(os.path.join(NEW_DATASET, "data.yaml"), "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

print("Cleaned dataset created:", NEW_DATASET)
print("Final classes:")
for i, cls in enumerate(FINAL_CLASSES):
    print(f"{i}: {cls}")