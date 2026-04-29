import argparse
import os

import nibabel as nib
import numpy as np
import torch
from torch.nn import DataParallel
from tqdm import tqdm

from config import Config, FINE_TUNING
from dataset import Dataset_BraTs19, Dataset_single
from models.CrossModalUNet import CrossModalUNet
from models.unet import UNet


def env_path(name, default):
    return os.environ.get(name, default)


def load_model(config, model_path, device):
    if config.model == "UNet":
        net_1 = UNet(config.num_classes, mode="seg")
        net_2 = UNet(config.num_classes, mode="seg")
    elif config.model == "CrossModalUNet":
        net_1 = CrossModalUNet(
            config.num_classes,
            in_channels=config.in_channels,
            mode="seg",
            num_modalities=2,
            growth_rate=config.growth_rate,
            use_anatomical_alignment=config.use_anatomical_alignment,
            use_global_local_loss=config.use_global_local_loss,
            baseline=config.baseline,
            use_cross_align=config.use_cross_align,
        )
        net_2 = None
    else:
        raise ValueError(f"Unknown model: {config.model}")

    checkpoint = torch.load(model_path, map_location=device)
    net_1 = DataParallel(net_1).to(device)
    net_1.load_state_dict(checkpoint["model1"])
    if net_2 is not None and "model2" in checkpoint:
        net_2 = DataParallel(net_2).to(device)
        net_2.load_state_dict(checkpoint["model2"])

    net_1.to(device).eval()
    if net_2 is not None:
        net_2.to(device).eval()

    return net_1, net_2


def test_brats19(model, config, device, result_dir):
    os.makedirs(result_dir, exist_ok=True)

    dataset = Dataset_BraTs19(config, training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=config.num_cpu_workers,
    )

    for batch in tqdm(loader, desc="Testing BraTS19"):
        mod1, mod2, label1, label2, patient_ids = batch
        mod1 = mod1.to(device)
        mod2 = mod2.to(device)

        with torch.no_grad():
            if model[1] is None:
                pred1, pred2 = model[0]([mod1, mod2])
            else:
                pred1 = model[0](mod1)
                pred2 = model[1](mod2)

        pred1 = torch.argmax(pred1, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred2 = torch.argmax(pred2, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        image_mod1 = mod1.squeeze().cpu().numpy()
        image_mod2 = mod2.squeeze().cpu().numpy()
        label = label1.squeeze().cpu().numpy().astype(np.uint8)

        for patient_id in patient_ids:
            patient_dir = os.path.join(result_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            affine = np.diag([1.5, 1.5, 1.5, 1.0])
            nib.save(nib.Nifti1Image(image_mod1, affine), os.path.join(patient_dir, f"{patient_id}_mod1_image.nii.gz"))
            nib.save(nib.Nifti1Image(image_mod2, affine), os.path.join(patient_dir, f"{patient_id}_mod2_image.nii.gz"))
            nib.save(nib.Nifti1Image(label, affine), os.path.join(patient_dir, f"{patient_id}_label.nii.gz"))
            nib.save(nib.Nifti1Image(pred1, affine), os.path.join(patient_dir, f"{patient_id}_mod1_pred.nii.gz"))
            nib.save(nib.Nifti1Image(pred2, affine), os.path.join(patient_dir, f"{patient_id}_mod2_pred.nii.gz"))


def apply_carotid_test_paths(config):
    config.image_dir_mod1_tr = env_path("DYNACOLLAB_CAROTID_CT_TEST_IMAGES", "./data/CarotidArtery_CT/imagesTs")
    config.label_dir_mod1_tr = env_path("DYNACOLLAB_CAROTID_CT_TEST_LABELS", "./data/CarotidArtery_CT/labelsTs")
    config.image_dir_mod2_tr = env_path("DYNACOLLAB_CAROTID_MRI_TEST_IMAGES", "./data/CarotidArtery_MRI/imagesTs")
    config.label_dir_mod2_tr = env_path("DYNACOLLAB_CAROTID_MRI_TEST_LABELS", "./data/CarotidArtery_MRI/labelsTs")


def test_carotid(model, config, device, result_dir):
    apply_carotid_test_paths(config)
    os.makedirs(result_dir, exist_ok=True)

    dataset = Dataset_single(config, training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=config.num_cpu_workers,
    )

    for batch in tqdm(loader, desc="Testing carotid"):
        mod1, mod2, label1, label2, sample_ids = batch
        mod1 = mod1.to(device)
        mod2 = mod2.to(device)

        with torch.no_grad():
            if model[1] is None:
                pred1, pred2 = model[0]([mod1, mod2])
            else:
                pred1 = model[0](mod1)
                pred2 = model[1](mod2)

        image_ct = mod1.squeeze().cpu().numpy()
        image_mri = mod2.squeeze().cpu().numpy()
        label_ct = label1.squeeze().cpu().numpy().astype(np.uint8)
        label_mri = label2.squeeze().cpu().numpy().astype(np.uint8)
        pred_ct = torch.argmax(pred1, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred_mri = torch.argmax(pred2, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        sample_id = sample_ids[0].item()

        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        nib.save(nib.Nifti1Image(image_ct, affine), os.path.join(result_dir, f"{sample_id}_CT_image.nii.gz"))
        nib.save(nib.Nifti1Image(label_ct, affine), os.path.join(result_dir, f"{sample_id}_CT_label.nii.gz"))
        nib.save(nib.Nifti1Image(pred_ct, affine), os.path.join(result_dir, f"{sample_id}_CT_pred.nii.gz"))
        nib.save(nib.Nifti1Image(image_mri, affine), os.path.join(result_dir, f"{sample_id}_MRI_image.nii.gz"))
        nib.save(nib.Nifti1Image(label_mri, affine), os.path.join(result_dir, f"{sample_id}_MRI_label.nii.gz"))
        nib.save(nib.Nifti1Image(pred_mri, affine), os.path.join(result_dir, f"{sample_id}_MRI_pred.nii.gz"))


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference and export NIfTI predictions.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--config", type=str, default="dongmai", choices=["BraTs19", "dongmai"], help="Dataset configuration.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = Config(FINE_TUNING, data=args.config)
    if args.config == "BraTs19":
        config.dir_tr = env_path("DYNACOLLAB_BRATS_VAL_DIR", "./data/BraTS19/VAL")

    model = load_model(config, args.model_path, device)
    model_name = os.path.basename(args.model_path).split(".")[0]
    result_dir = args.output_dir or os.path.join("test_result", model_name)
    os.makedirs(result_dir, exist_ok=True)

    if args.config == "BraTs19":
        test_brats19(model, config, device, os.path.join(result_dir, "BraTS19"))
    else:
        test_carotid(model, config, device, result_dir)


if __name__ == "__main__":
    main()
