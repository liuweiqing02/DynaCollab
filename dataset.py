import logging
import os
import random
import re

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset, random_split

from augmentations import Blur, Crop, Cutout, Flip, Noise, Normalize, SafeTransformer

logger = logging.getLogger("DynaCollab")


def load_nifti_file(file_path, num_classes=None):
    img = nib.load(file_path)
    data = img.get_fdata()
    original_spacing = img.header.get_zooms()[:3]
    affine = img.affine

    if "label" in file_path.lower():
        data = (data > 0).astype(data.dtype)

    return data, original_spacing, affine


def load_nifti_file_brains(file_path, num_classes=None):
    img = nib.load(file_path)
    data = img.get_fdata()
    original_spacing = img.header.get_zooms()[:3]
    affine = img.affine

    if "seg" in file_path.lower() and num_classes is not None:
        new_data = np.zeros_like(data, dtype=np.uint8)
        new_data[data == 0] = 0
        new_data[data == 1] = 1
        new_data[data == 2] = 2
        new_data[data == 4] = 3
        data = new_data

    return data, original_spacing, affine


def resample_data(data, original_spacing, desired_spacing, is_label=False):
    resize_factor = np.array(original_spacing) / np.array(desired_spacing)
    new_shape = np.round(np.array(data.shape) * resize_factor)
    real_resize_factor = new_shape / np.array(data.shape)
                                                             
    order = 0 if is_label else 1
    return zoom(data, real_resize_factor, mode="nearest", order=order)


def preprocess_data_3shape(data, target_shape, method="pad"):
    width, depth, height = data.shape
    target_width, target_depth, target_height = target_shape

    if method == "pad":
        if width < target_width:
            pad = target_width - width
            data = np.pad(data, ((pad // 2, pad - pad // 2), (0, 0), (0, 0)), mode="constant")
        elif width > target_width:
            start = (width - target_width) // 2
            data = data[start : start + target_width, :, :]

        if depth < target_depth:
            pad = target_depth - depth
            data = np.pad(data, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), mode="constant")
        elif depth > target_depth:
            start = (depth - target_depth) // 2
            data = data[:, start : start + target_depth, :]

        if height < target_height:
            pad = target_height - height
            data = np.pad(data, ((0, 0), (0, 0), (pad // 2, pad - pad // 2)), mode="constant")
        elif height > target_height:
            start = (height - target_height) // 2
            data = data[:, :, start : start + target_height]

    elif method == "resize":
        zoom_factors = [target_width / width, target_depth / depth, target_height / height]
        data = zoom(data, zoom_factors, order=1)

    return data


def fixed_center_crop(arr, target_shape=(80, 80, 80)):
    orig_d, orig_h, orig_w = arr.shape
    target_d, target_h, target_w = target_shape
    assert orig_d >= target_d and orig_h >= target_h and orig_w >= target_w

    start_d = (orig_d - target_d) // 2
    start_h = (orig_h - target_h) // 2
    start_w = (orig_w - target_w) // 2
    return arr[start_d : start_d + target_d, start_h : start_h + target_h, start_w : start_w + target_w]


def extract_digit_from_filename(filename):
    digits = re.findall(r"\d+", filename)
    return digits[0] if digits else None


def extract_patient_id(folder_name):
    if folder_name.startswith("BraTS19_"):
        base_name = folder_name[8:]
    else:
        base_name = folder_name
    parts = base_name.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return base_name


def process_lists_to_collect_digits(lists):
    collected_digits = []
    for tuple_items in zip(*lists):
        digits = [extract_digit_from_filename(item) for item in tuple_items]
        if len(set(digits)) != 1:
            return []
        collected_digits.append(digits[0])
    return collected_digits


def get_unique_labels(label):
    unique_labels, counts = np.unique(label.flatten(), return_counts=True)
    for label_value, count in zip(unique_labels, counts):
        print(f"Label {label_value}: {count} pixels")


def _get_spatial_shape(input_size):
    if len(input_size) == 4:
        return np.array(input_size[1:], dtype=np.int32)
    return np.array(input_size, dtype=np.int32)


def build_vessel_safe_aug(config, input_size):
    transformer = SafeTransformer()
    spatial_shape = _get_spatial_shape(input_size)

    if config.tf == "no_tf":
        transformer.register(Normalize(), 1.0, "image")
    elif config.tf == "all_tf":
        transformer.register(Normalize(), 1.0, "image")
        transformer.register(Blur(sigma=(0.1, 0.5)), 0.3, "image")
        transformer.register(Noise(sigma=(0.1, 0.5)), 0.3, "image")
        transformer.register(Cutout(patch_size=np.ceil(spatial_shape / 6).astype(np.int32)), 0.3, "image")
        transformer.register(Flip(), 0.5, "both")
        transformer.register(Crop(np.ceil(0.8 * spatial_shape).astype(np.int32), "random", resize=True), 0.5, "both")

    return transformer


def build_vessel_safe_aug_val(config, input_size):
    transformer = SafeTransformer()
    transformer.register(Normalize(), 1.0, "image")
    return transformer


class DatasetCarotid(Dataset):
    def __init__(self, config, training=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.training = training

        self.enable_memory_cache = getattr(config, "enable_memory_cache", True)
        self.mod1_cache = {}
        self.mod2_cache = {}

        self.transforms = build_vessel_safe_aug(config, config.input_size)
        self.transforms_val = build_vessel_safe_aug_val(config, config.input_size)

        self.image_dir_mod1 = config.image_dir_mod1_tr
        self.label_dir_mod1 = config.label_dir_mod1_tr
        self.image_dir_mod2 = config.image_dir_mod2_tr
        self.label_dir_mod2 = config.label_dir_mod2_tr
        self.target_size = config.target_size
        self.target_size_crop = config.target_size_crop
        self.desired_spacing = config.desired_spacing

        self.image_paths_mod1 = sorted([f for f in os.listdir(self.image_dir_mod1) if f.endswith(".nii.gz")])
        self.label_paths_mod1 = sorted([f for f in os.listdir(self.label_dir_mod1) if f.endswith(".nii.gz")])
        self.image_paths_mod2 = sorted([f for f in os.listdir(self.image_dir_mod2) if f.endswith(".nii.gz")])
        self.label_paths_mod2 = sorted([f for f in os.listdir(self.label_dir_mod2) if f.endswith(".nii.gz")])

        assert len(self.image_paths_mod1) == len(self.label_paths_mod1)
        assert len(self.image_paths_mod2) == len(self.label_paths_mod2)

        all_lists = [self.image_paths_mod1, self.label_paths_mod1, self.image_paths_mod2, self.label_paths_mod2]
        collected_digits = process_lists_to_collect_digits(all_lists)
        if not collected_digits:
            raise ValueError("Modality/image/label filename IDs are not aligned across folders.")
        logger.info(f"DatasetCarotid paired samples: {len(collected_digits)}")

    def collate_fn(self, list_samples):
        list_mod1 = torch.stack(
            [torch.as_tensor(mod1, dtype=torch.float) for (mod1, mod2, mod1_label, mod2_label, id_val) in list_samples], dim=0
        )
        list_mod2 = torch.stack(
            [torch.as_tensor(mod2, dtype=torch.float) for (mod1, mod2, mod1_label, mod2_label, id_val) in list_samples], dim=0
        )
        list_mod1_label = torch.stack(
            [torch.as_tensor(mod1_label, dtype=torch.float) for (mod1, mod2, mod1_label, mod2_label, id_val) in list_samples], dim=0
        )
        list_mod2_label = torch.stack(
            [torch.as_tensor(mod2_label, dtype=torch.float) for (mod1, mod2, mod1_label, mod2_label, id_val) in list_samples], dim=0
        )
        list_id = torch.stack(
            [torch.as_tensor(id_val, dtype=torch.long) for (mod1, mod2, mod1_label, mod2_label, id_val) in list_samples], dim=0
        )
        return list_mod1, list_mod2, list_mod1_label, list_mod2_label, list_id

    def __getitem__(self, idx):
        try:
            if self.enable_memory_cache and idx in self.mod2_cache:
                mod1_data = self.mod1_cache[idx].copy()
                mod2_data = self.mod2_cache[idx].copy()
            else:
                mod1_image_path = os.path.join(self.image_dir_mod1, self.image_paths_mod1[idx])
                mod1_image, original_spacing, _ = load_nifti_file(mod1_image_path)
                mod1_image = resample_data(mod1_image, original_spacing, self.desired_spacing, is_label=False)
                mod1_image = preprocess_data_3shape(mod1_image, self.target_size, method="pad")
                mod1_image = fixed_center_crop(mod1_image, self.target_size_crop)
                mod1_image = np.expand_dims(mod1_image, axis=0)

                mod2_image_path = os.path.join(self.image_dir_mod2, self.image_paths_mod2[idx])
                mod2_image, original_spacing, _ = load_nifti_file(mod2_image_path)
                mod2_image = resample_data(mod2_image, original_spacing, self.desired_spacing, is_label=False)
                mod2_image = preprocess_data_3shape(mod2_image, self.target_size, method="pad")
                mod2_image = fixed_center_crop(mod2_image, self.target_size_crop)
                mod2_image = np.expand_dims(mod2_image, axis=0)

                mod1_label_path = os.path.join(self.label_dir_mod1, self.label_paths_mod1[idx])
                mod2_label_path = os.path.join(self.label_dir_mod2, self.label_paths_mod2[idx])

                mod1_label, original_spacing, _ = load_nifti_file(mod1_label_path, num_classes=self.config.num_classes)
                mod1_label = resample_data(mod1_label, original_spacing, self.desired_spacing, is_label=True)
                mod1_label = preprocess_data_3shape(mod1_label, self.target_size, method="pad")
                mod1_label = fixed_center_crop(mod1_label, self.target_size_crop)
                mod1_label = np.expand_dims(mod1_label, axis=0)

                mod2_label, original_spacing, _ = load_nifti_file(mod2_label_path, num_classes=self.config.num_classes)
                mod2_label = resample_data(mod2_label, original_spacing, self.desired_spacing, is_label=True)
                mod2_label = preprocess_data_3shape(mod2_label, self.target_size, method="pad")
                mod2_label = fixed_center_crop(mod2_label, self.target_size_crop)
                mod2_label = np.expand_dims(mod2_label, axis=0)

                mod1_data = np.concatenate((mod1_image, mod1_label), axis=0)
                mod2_data = np.concatenate((mod2_image, mod2_label), axis=0)

                if self.enable_memory_cache:
                    self.mod1_cache[idx] = mod1_data.copy()
                    self.mod2_cache[idx] = mod2_data.copy()

            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            mod1_data = (self.transforms if self.training else self.transforms_val)(mod1_data).copy()

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            mod2_data = (self.transforms if self.training else self.transforms_val)(mod2_data).copy()

            mod1_data_copy = mod1_data.copy()
            mod2_data_copy = mod2_data.copy()
            mod1_data = np.expand_dims(mod1_data_copy[0], axis=0)
            mod2_data = np.expand_dims(mod2_data_copy[0], axis=0)
            mod1_label = np.expand_dims(mod1_data_copy[1], axis=0)
            mod2_label = np.expand_dims(mod2_data_copy[1], axis=0)

            id_extracted = int(extract_digit_from_filename(self.image_paths_mod1[idx]))
            return mod1_data, mod2_data, mod1_label, mod2_label, id_extracted

        except Exception as e:
            logger.error(f"Error at index {idx}: {e}")
            raise

    def __len__(self):
        return len(self.image_paths_mod1)

    def get_id(self, idx):
        return int(extract_digit_from_filename(self.image_paths_mod1[idx]))


class DatasetBraTS19(Dataset):
    def __init__(self, config, training=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.training = training
        self.num_classes = config.num_classes
        self.enable_memory_cache = getattr(config, "enable_memory_cache", True)

        self.mod1_cache = {}
        self.mod2_cache = {}
        self.label_cache = {}

        self.transforms = build_vessel_safe_aug(config, config.input_size)
        self.transforms_val = build_vessel_safe_aug_val(config, config.input_size)

        self.data_dir = config.dir_tr
        self.target_size = config.target_size
        self.target_size_crop = config.target_size_crop
        self.desired_spacing = config.desired_spacing

        self.patient_folders = []
        self.file_prefixes = {}

        for folder in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder)
            if not os.path.isdir(folder_path):
                continue

            possible_prefixes = [folder, "_".join(folder.split("_")[:2])]
            valid_folder = False
            for prefix in possible_prefixes:
                required_files = [
                    f"{prefix}_t1.nii",
                    f"{prefix}_t1ce.nii",
                    f"{prefix}_t2.nii",
                    f"{prefix}_flair.nii",
                    f"{prefix}_seg.nii",
                ]
                if all(os.path.exists(os.path.join(folder_path, req)) for req in required_files):
                    self.patient_folders.append(folder)
                    self.file_prefixes[folder] = prefix
                    valid_folder = True
                    break
            if not valid_folder:
                logger.warning(f"Skipping invalid BraTS folder: {folder}")

        if not self.patient_folders:
            raise RuntimeError(f"No valid BraTS19 cases found in {self.data_dir}")

        logger.info(f"DatasetBraTS19 valid cases: {len(self.patient_folders)}")

    def _load_image_channel(self, file_path):
        image, spacing, _ = load_nifti_file_brains(file_path)
        image = resample_data(image, spacing, self.desired_spacing, is_label=False)
        image = preprocess_data_3shape(image, self.target_size, method="pad")
        image = fixed_center_crop(image, self.target_size_crop)
        return np.expand_dims(image, axis=0)

    def collate_fn(self, list_samples):
        list_mod1, list_mod2, list_label1, list_label2, list_id = [], [], [], [], []
        for mod1, mod2, label1, label2, id_str in list_samples:
            list_mod1.append(torch.as_tensor(mod1, dtype=torch.float))
            list_mod2.append(torch.as_tensor(mod2, dtype=torch.float))
            list_label1.append(torch.as_tensor(label1, dtype=torch.float))
            list_label2.append(torch.as_tensor(label2, dtype=torch.float))
            list_id.append(id_str)
        return (
            torch.stack(list_mod1, dim=0),
            torch.stack(list_mod2, dim=0),
            torch.stack(list_label1, dim=0),
            torch.stack(list_label2, dim=0),
            list_id,
        )

    def __getitem__(self, idx):
        try:
            patient_folder = self.patient_folders[idx]
            folder_path = os.path.join(self.data_dir, patient_folder)
            patient_id = extract_patient_id(patient_folder)
            prefix = self.file_prefixes[patient_folder]

            if self.enable_memory_cache and idx in self.mod1_cache:
                mod1_image = self.mod1_cache[idx].copy()
                mod2_image = self.mod2_cache[idx].copy()
                label_data = self.label_cache[idx].copy()
            else:
                t1_path = os.path.join(folder_path, f"{prefix}_t1.nii")
                t1ce_path = os.path.join(folder_path, f"{prefix}_t1ce.nii")
                t2_path = os.path.join(folder_path, f"{prefix}_t2.nii")
                flair_path = os.path.join(folder_path, f"{prefix}_flair.nii")
                label_path = os.path.join(folder_path, f"{prefix}_seg.nii")

                t1 = self._load_image_channel(t1_path)
                t1ce = self._load_image_channel(t1ce_path)
                t2 = self._load_image_channel(t2_path)
                flair = self._load_image_channel(flair_path)

                label, label_spacing, _ = load_nifti_file_brains(label_path, num_classes=self.num_classes)
                label = resample_data(label, label_spacing, self.desired_spacing, is_label=True)
                label = preprocess_data_3shape(label, self.target_size, method="pad")
                label = fixed_center_crop(label, self.target_size_crop)
                label = np.expand_dims(label, axis=0)

                mod1_image = np.concatenate((t1, t1ce), axis=0)
                mod2_image = np.concatenate((t2, flair), axis=0)
                label_data = label
                if self.enable_memory_cache:
                    self.mod1_cache[idx] = mod1_image.copy()
                    self.mod2_cache[idx] = mod2_image.copy()
                    self.label_cache[idx] = label.copy()

            mod1_data = np.concatenate((mod1_image, label_data), axis=0)
            mod2_data = np.concatenate((mod2_image, label_data), axis=0)

            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            mod1_data = (self.transforms if self.training else self.transforms_val)(mod1_data).copy()

            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            mod2_data = (self.transforms if self.training else self.transforms_val)(mod2_data).copy()

            mod1_data_copy = mod1_data.copy()
            mod2_data_copy = mod2_data.copy()
            mod1_data = mod1_data_copy[:-1]
            mod2_data = mod2_data_copy[:-1]
            mod1_label = np.expand_dims(mod1_data_copy[-1], axis=0)
            mod2_label = np.expand_dims(mod2_data_copy[-1], axis=0)
            return mod1_data, mod2_data, mod1_label, mod2_label, patient_id

        except Exception as e:
            logger.error(f"Error processing patient {self.patient_folders[idx]}: {e}")
            raise

    def __len__(self):
        return len(self.patient_folders)

    def get_id(self, idx):
        return extract_patient_id(self.patient_folders[idx])


def split_dataset(dataset, train_ratio=0.8, seed=42):
    assert 0 < train_ratio < 1, "train_ratio must be in (0, 1)"
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    return random_split(dataset, [train_size, val_size], generator=generator)
