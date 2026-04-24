import os
import argparse
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from dataset import Dataset_BraTs19, Dataset_single
from models.unet import UNet
from models.CrossModalUNet import CrossModalUNet
from config import Config, FINE_TUNING
from torch.nn import DataParallel

def load_model(config, model_path, device):
    """加载训练好的模型"""
    # 根据配置初始化模型
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

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    net_1 = DataParallel(net_1).to(device)
    net_1.load_state_dict(checkpoint['model1'])
    if net_2 is not None and 'model2' in checkpoint:
        net_2 = DataParallel(net_2).to(device)
        net_2.load_state_dict(checkpoint['model2'])

    net_1.to(device).eval()
    if net_2 is not None:
        net_2.to(device).eval()

    return net_1, net_2


def test_braTs19(model, config, device, result_dir):
    """在BraTs19验证集上测试模型"""
    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)

    # 加载数据集
    dataset = Dataset_BraTs19(config, training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=config.num_cpu_workers
    )

    # 遍历数据集进行预测
    for batch in tqdm(loader, desc="Testing BraTs19"):
        mod1, mod2, label1, label2, patient_ids = batch
        mod1 = mod1.to(device)
        mod2 = mod2.to(device)

        with torch.no_grad():
            if model[1] is None:  # 单流模型
                pred1, pred2 = model[0]([mod1, mod2])
            else:  # 双流模型
                pred1 = model[0](mod1)
                pred2 = model[1](mod2)

        # 转换为numpy数组
        pred1 = torch.argmax(pred1, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred2 = torch.argmax(pred2, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        image_mod1 = mod1.squeeze().cpu().numpy()
        image_mod2 = mod2.squeeze().cpu().numpy()
        label = label1.squeeze().cpu().numpy().astype(np.uint8)

        # 保存结果
        for i, patient_id in enumerate(patient_ids):
            # 创建患者专属目录
            patient_dir = os.path.join(result_dir, patient_id)
            os.makedirs(patient_dir, exist_ok=True)

            # 保存NIfTI文件
            affine = np.diag([1.5, 1.5, 1.5, 1.0])  # 默认仿射矩阵
            nib.save(nib.Nifti1Image(image_mod1, affine),
                     os.path.join(patient_dir, f"{patient_id}_mod1_image.nii.gz"))
            nib.save(nib.Nifti1Image(image_mod2, affine),
                     os.path.join(patient_dir, f"{patient_id}_mod2_image.nii.gz"))
            nib.save(nib.Nifti1Image(label, affine),
                     os.path.join(patient_dir, f"{patient_id}_label.nii.gz"))
            nib.save(nib.Nifti1Image(pred1, affine),
                     os.path.join(patient_dir, f"{patient_id}_mod1_pred.nii.gz"))
            nib.save(nib.Nifti1Image(pred2, affine),
                     os.path.join(patient_dir, f"{patient_id}_mod2_pred.nii.gz"))


# def test_dongmai(model, config, device, result_dir, modality):
#     """在dongmai验证集上测试模型"""
#     # 根据模态设置路径
#     if modality == "CT":
#         config.image_dir_mod1_tr = "../data_125/Dataset004_dongmaiCT/imagesTs"
#         config.label_dir_mod1_tr = "../data_125/Dataset004_dongmaiCT/labelsTs"
#         config.image_dir_mod2_tr = "../data_125/Dataset004_dongmaiCT/imagesTs"  # 未使用
#         config.label_dir_mod2_tr = "../data_125/Dataset004_dongmaiCT/labelsTs"  # 未使用
#     elif modality == "MR":
#         config.image_dir_mod1_tr = "../data_125/Dataset005_dongmaiMR/imagesTs"
#         config.label_dir_mod1_tr = "../data_125/Dataset005_dongmaiMR/labelsTs"
#         config.image_dir_mod2_tr = "../data_125/Dataset005_dongmaiMR/imagesTs"  # 未使用
#         config.label_dir_mod2_tr = "../data_125/Dataset005_dongmaiMR/labelsTs"  # 未使用
#
#     # 创建结果目录
#     modality_dir = os.path.join(result_dir, modality)
#     os.makedirs(modality_dir, exist_ok=True)
#
#     # 加载数据集
#     dataset = Dataset_single(config, training=False)
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         collate_fn=dataset.collate_fn,
#         num_workers=config.num_cpu_workers
#     )
#
#     # 遍历数据集进行预测
#     for batch in tqdm(loader, desc=f"Testing dongmai {modality}"):
#         mod1, mod2, label1, label2, sample_ids = batch
#         mod1 = mod1.to(device)
#         mod2 = mod2.to(device)
#
#         with torch.no_grad():
#             if model[1] is None:  # 单流模型
#                 pred1, pred2 = model[0]([mod1, mod2])
#             else:  # 双流模型
#                 pred1 = model[0](mod1)
#                 pred2 = model[1](mod2)
#
#         # 转换为numpy数组
#         pred = torch.argmax(pred1 if modality == "CT" else pred2,
#                             dim=1).squeeze().cpu().numpy().astype(np.uint8)
#         image = mod1.squeeze().cpu().numpy()
#         label = label1.squeeze().cpu().numpy().astype(np.uint8)
#         sample_id = sample_ids[0].item()
#
#         # 保存结果
#         affine = np.diag([1.5, 1.5, 1.5, 1.0])  # 默认仿射矩阵
#         nib.save(nib.Nifti1Image(image, affine),
#                  os.path.join(modality_dir, f"{sample_id}_image.nii.gz"))
#         nib.save(nib.Nifti1Image(label, affine),
#                  os.path.join(modality_dir, f"{sample_id}_label.nii.gz"))
#         nib.save(nib.Nifti1Image(pred, affine),
#                  os.path.join(modality_dir, f"{sample_id}_pred.nii.gz"))


def test_dongmai(model, config, device, result_dir):
    """测试dongmai数据集的双模态模型"""
    # 设置双模态数据路径
    config.image_dir_mod1_tr = "../data_125/Dataset004_dongmaiCT/imagesTs"
    config.label_dir_mod1_tr = "../data_125/Dataset004_dongmaiCT/labelsTs"
    config.image_dir_mod2_tr = "../data_125/Dataset005_dongmaiMR/imagesTs"
    config.label_dir_mod2_tr = "../data_125/Dataset005_dongmaiMR/labelsTs"

    # 创建结果目录
    os.makedirs(result_dir, exist_ok=True)

    # 加载数据集
    dataset = Dataset_single(config, training=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=config.num_cpu_workers
    )

    # 遍历数据集进行预测
    for batch in tqdm(loader, desc="Testing dongmai"):
        mod1, mod2, label1, label2, sample_ids = batch
        mod1 = mod1.to(device)
        mod2 = mod2.to(device)

        with torch.no_grad():
            if model[1] is None:  # 单流模型
                pred1, pred2 = model[0]([mod1, mod2])
            else:  # 双流模型
                pred1 = model[0](mod1)
                pred2 = model[1](mod2)

        # 转换为numpy数组
        image_ct = mod1.squeeze().cpu().numpy()
        image_mr = mod2.squeeze().cpu().numpy()
        label_ct = label1.squeeze().cpu().numpy().astype(np.uint8)
        label_mr = label2.squeeze().cpu().numpy().astype(np.uint8)
        pred_ct = torch.argmax(pred1, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        pred_mr = torch.argmax(pred2, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        sample_id = sample_ids[0].item()

        # 保存结果 - 使用原始文件名
        affine = np.diag([1.5, 1.5, 1.5, 1.0])  # 默认仿射矩阵

        # 保存CT模态结果
        nib.save(nib.Nifti1Image(image_ct, affine),
                 os.path.join(result_dir, f"{sample_id}_CT_image.nii.gz"))
        nib.save(nib.Nifti1Image(label_ct, affine),
                 os.path.join(result_dir, f"{sample_id}_CT_label.nii.gz"))
        nib.save(nib.Nifti1Image(pred_ct, affine),
                 os.path.join(result_dir, f"{sample_id}_CT_pred.nii.gz"))

        # 保存MR模态结果
        nib.save(nib.Nifti1Image(image_mr, affine),
                 os.path.join(result_dir, f"{sample_id}_MR_image.nii.gz"))
        nib.save(nib.Nifti1Image(label_mr, affine),
                 os.path.join(result_dir, f"{sample_id}_MR_label.nii.gz"))
        nib.save(nib.Nifti1Image(pred_mr, affine),
                 os.path.join(result_dir, f"{sample_id}_MR_pred.nii.gz"))


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Test trained model on validation sets")
    parser.add_argument("--model_path", type=str, default="./classification_checkpoint_dir/DM_FT_MV5_nofztf_1e-3_3_1.5_0.2(PT0.4,0.6_FT0.2,0.8)_best.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--config", type=str, default="dongmai",
                        choices=["BraTs19", "dongmai"],
                        help="Dataset configuration to use")
    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建配置对象
    config = Config(FINE_TUNING)
    config.data = args.config

    # 加载模型
    net_1, net_2 = load_model(config, args.model_path, device)
    model = (net_1, net_2)

    # 创建结果目录
    model_name = os.path.basename(args.model_path).split('.')[0]
    result_dir = os.path.join("test_result", model_name)
    os.makedirs(result_dir, exist_ok=True)

    # 在BraTs19验证集上测试
    if args.config == "BraTs19":
        config.dir_tr = "../BraTs19/VAL"
        test_braTs19(model, config, device, os.path.join(result_dir, "BraTs19"))

    # 在dongmai验证集上测试
    elif args.config == "dongmai":
        test_dongmai(model, config, device, result_dir)


if __name__ == "__main__":
    main()
