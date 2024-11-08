import random
import shutil
import yaml
import os


def prepare_dataset(ikdataset, dataset_folder, split_ratio):
    if _dataset_exists(ikdataset, dataset_folder, split_ratio):
        return dataset_folder + os.sep + "dataset.yaml"

    train_img_folder = dataset_folder + os.sep + "images" + os.sep + "train"
    val_img_folder = dataset_folder + os.sep + "images" + os.sep + "val"
    train_label_folder = dataset_folder + os.sep + "labels" + os.sep + "train"
    val_label_folder = dataset_folder + os.sep + "labels" + os.sep + "val"
    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)

    images = ikdataset.data["images"]
    val_size = int((1-split_ratio) * len(images))
    val_indices = random.sample(range(len(images)), k=val_size)
    index = 0

    for img in images:
        src_filename = img["filename"]

        if index in val_indices:
            dst_filename = val_img_folder + os.sep + \
                os.path.basename(src_filename)
            shutil.copy(src_filename, dst_filename)
            dst_filename = dst_filename.replace(
                "images" + os.sep + "val", "labels" + os.sep + "val", 1)
            dst_filename = dst_filename.replace(
                '.' + dst_filename.split('.')[-1], '.txt')
            _create_image_labels(
                dst_filename, img["annotations"], img["width"], img["height"])
        else:
            dst_filename = train_img_folder + \
                os.sep + os.path.basename(src_filename)
            shutil.copy(src_filename, dst_filename)
            dst_filename = dst_filename.replace(
                "images" + os.sep + "train", "labels" + os.sep + "train", 1)
            dst_filename = dst_filename.replace(
                '.' + dst_filename.split('.')[-1], '.txt')
            _create_image_labels(
                dst_filename, img["annotations"], img["width"], img["height"])

        index += 1

    categories = ikdataset.data["metadata"]["category_names"]
    return _create_dataset_yaml(dataset_folder, train_img_folder, val_img_folder, categories)


def _dataset_exists(ikdataset, dataset_folder, split_ratio):
    dataset_yaml = dataset_folder + os.sep + "dataset.yaml"

    if not os.path.exists(dataset_yaml):
        return False

    # Dataset exist
    with open(dataset_yaml, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

        # Check folder structure
        if not os.path.exists(data["train"]) or not os.path.exists(data["val"]):
            f.close()
            shutil.rmtree(dataset_folder)
            return False

        train_label_path = data["train"].replace(os.sep + "images" + os.sep + "train",
                                                 os.sep + "labels" + os.sep + "train",
                                                 1)
        val_label_path = data["val"].replace(os.sep + "images" + os.sep + "val",
                                             os.sep + "labels" + os.sep + "val",
                                             1)

        if not os.path.exists(train_label_path) or not os.path.exists(val_label_path):
            f.close()
            shutil.rmtree(dataset_folder)
            return False

        # check number of classes
        categories = ikdataset.data["metadata"]["category_names"]

        if len(categories) != data["nc"]:
            f.close()
            shutil.rmtree(dataset_folder)
            return False

        if len(categories) != len(data["names"]):
            f.close()
            shutil.rmtree(dataset_folder)
            return False

        # check number of images and labels
        images = ikdataset.data["images"]
        val_size = int((1 - split_ratio) * len(images))
        train_size = len(images) - val_size

        train_images_count = len(os.listdir(data["train"]))
        train_labels_count = len(os.listdir(train_label_path))

        if train_images_count != train_size or train_labels_count != train_size:
            f.close()
            shutil.rmtree(dataset_folder)
            return False

        val_images_count = len(os.listdir(data["val"]))
        val_labels_count = len(os.listdir(val_label_path))

        if val_images_count != val_size or val_labels_count != val_size:
            f.close()
            shutil.rmtree(dataset_folder)
            return False

    print("A valid YoloV5 dataset structure already exists, skip building a new one")
    return True


def _create_image_labels(filename, annotations, img_w, img_h):
    with open(filename, "w+") as f:
        for ann in annotations:
            f.write('%d ' % ann["category_id"])
            if len(ann["segmentation_poly"]):
                # this format does not support having several polygons for a single object
                for i, coord in enumerate(ann["segmentation_poly"][0]):
                    if i % 2 == 0:
                        coord /= img_w
                    else:
                        coord /= img_h
                    if i < len(ann["segmentation_poly"][0]) - 1:
                        f.write('%f ' % coord)
                    else:
                        f.write('%f\n' % coord)


def _create_dataset_yaml(dataset_folder, train_folder, val_folder, categories):
    dataset = {"train": train_folder,
               "val": val_folder,
               "nc": len(categories),
               "names": list(categories.values())}

    dataset_yaml_file = dataset_folder + os.sep + "dataset.yaml"
    with open(dataset_yaml_file, "w") as f:
        yaml.dump(dataset, f, default_flow_style=True, sort_keys=False)

    return dataset_yaml_file
