import numpy as np
import torch
import torch.nn.functional as F


class SegmentationMetrics:
    def __init__(self, num_classes=2, ignore_background=True, spacing=(1.5, 1.5, 1.5), eps=1e-6):
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.spacing = spacing        
        self.eps = eps
        self.dice_values = []
        self.iou_values = []
        self.hd95_values = []              

    def reset(self):
        """重置指标收集器"""
        self.dice_values = []
        self.iou_values = []
        self.hd95_values = []          

    def __call__(self, pred, target):
        """
        计算当前批次的 Dice、IoU 和 HD95 指标
        返回: (dice_per_class, iou_per_class, hd95_per_class) 形状为 [B, C]
        """
        pred = F.softmax(pred, dim=1)
        pred_labels = torch.argmax(pred, dim=1)

                       
        pred_onehot = F.one_hot(pred_labels, self.num_classes).permute(0, 4, 1, 2, 3).float()
        target_onehot = F.one_hot(target, self.num_classes).permute(0, 4, 1, 2, 3).float()

        if self.ignore_background:
            pred_onehot = pred_onehot[:, 1:, ...]
            target_onehot = target_onehot[:, 1:, ...]
            num_classes = self.num_classes - 1
        else:
            num_classes = self.num_classes

                        
        intersection = (pred_onehot * target_onehot).sum(dim=(2, 3, 4))
        pred_area = pred_onehot.sum(dim=(2, 3, 4))
        target_area = target_onehot.sum(dim=(2, 3, 4))
        union = pred_area + target_area - intersection

                                   
        dice_per_class = (2. * intersection + self.eps) / (pred_area + target_area + self.eps)
        iou_per_class = (intersection + self.eps) / (union + self.eps)

                               
        hd95_per_class = torch.zeros_like(dice_per_class)
        for c in range(num_classes):
            for b in range(pred_onehot.shape[0]):
                pred_mask = pred_onehot[b, c].cpu().numpy().astype(np.uint8)
                true_mask = target_onehot[b, c].cpu().numpy().astype(np.uint8)

                         
                if np.sum(true_mask) == 0 and np.sum(pred_mask) == 0:
                    hd95_val = 0.0
                elif np.sum(true_mask) == 0 or np.sum(pred_mask) == 0:
                                  
                    hd95_val = 373.0
                else:
                    try:
                        hd95_val = hd95(pred_mask, true_mask, voxelspacing=self.spacing)
                    except:
                        hd95_val = 373.0

                hd95_per_class[b, c] = hd95_val

        return dice_per_class, iou_per_class, hd95_per_class

    def update(self, dice_per_class, iou_per_class, hd95_per_class):
        """更新指标收集器"""
                           
        dice_per_sample = dice_per_class.mean(dim=1).cpu().numpy()
        iou_per_sample = iou_per_class.mean(dim=1).cpu().numpy()
        hd95_per_sample = hd95_per_class.mean(dim=1).cpu().numpy()              

        self.dice_values.extend(dice_per_sample)
        self.iou_values.extend(iou_per_sample)
        self.hd95_values.extend(hd95_per_sample)           

    def compute_stats(self):
        """计算所有收集样本的统计信息"""

        def _compute(arr):
            if len(arr) == 0:
                return 0.0, 0.0, "0.000 ± 0.000"
            mean = np.mean(arr)
            std = np.std(arr)
            return mean, std, f"{mean:.4f} ± {std:.4f}"

        dice_mean, dice_std, dice_str = _compute(self.dice_values)
        iou_mean, iou_std, iou_str = _compute(self.iou_values)
        hd95_mean, hd95_std, hd95_str = _compute(self.hd95_values)             

        return {
            "dice": {"mean": dice_mean, "std": dice_std, "str": dice_str},
            "iou": {"mean": iou_mean, "std": iou_std, "str": iou_str},
            "hd95": {"mean": hd95_mean, "std": hd95_std, "str": hd95_str}             
        }

            
import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import hd95                                 


class BraTSMetrics:
    def __init__(self, spacing=(1.5, 1.5, 1.5), eps=1e-6):
        self.spacing = spacing        
        self.eps = eps
                      
        self.et_dice = []
        self.tc_dice = []
        self.wt_dice = []
                     
        self.et_iou = []
        self.tc_iou = []
        self.wt_iou = []
                         
        self.et_hd95 = []
        self.tc_hd95 = []
        self.wt_hd95 = []

    def reset(self):
        """重置所有指标收集器"""
        self.et_dice = []
        self.tc_dice = []
        self.wt_dice = []
        self.et_iou = []
        self.tc_iou = []
        self.wt_iou = []
                   
        self.et_hd95 = []
        self.tc_hd95 = []
        self.wt_hd95 = []

    def __call__(self, pred, target):
        """
        计算当前批次的区域指标
        返回: 三个区域的Dice、IoU和HD95 (ET, TC, WT)
        """
        pred = F.softmax(pred, dim=1)
        pred_labels = torch.argmax(pred, dim=1)                

                
        et_pred = (pred_labels == 3).float()           
        tc_pred = torch.logical_or(pred_labels == 1, pred_labels == 3).float()             
        wt_pred = (pred_labels > 0).float()                     

        et_true = (target == 3).float()
        tc_true = torch.logical_or(target == 1, target == 3).float()
        wt_true = (target > 0).float()

                         
        et_dice, et_iou = self._compute_metrics(et_pred, et_true)
        tc_dice, tc_iou = self._compute_metrics(tc_pred, tc_true)
        wt_dice, wt_iou = self._compute_metrics(wt_pred, wt_true)

                        
        et_hd95 = self._compute_hd95(et_pred, et_true)
        tc_hd95 = self._compute_hd95(tc_pred, tc_true)
        wt_hd95 = self._compute_hd95(wt_pred, wt_true)

        return (et_dice, tc_dice, wt_dice), (et_iou, tc_iou, wt_iou), (et_hd95, tc_hd95, wt_hd95)

    def _compute_metrics(self, pred_mask, true_mask):
        """计算单个区域的Dice和IoU"""
        intersection = (pred_mask * true_mask).sum(dim=(1, 2, 3))
        pred_area = pred_mask.sum(dim=(1, 2, 3))
        true_area = true_mask.sum(dim=(1, 2, 3))
        union = pred_area + true_area - intersection

        dice = (2. * intersection + self.eps) / (pred_area + true_area + self.eps)
        iou = (intersection + self.eps) / (union + self.eps)

        return dice, iou

    def _compute_hd95(self, pred_mask, true_mask):
        """计算单个区域的HD95"""
        batch_size = pred_mask.shape[0]
        hd95_vals = torch.zeros(batch_size, device=pred_mask.device)

        for i in range(batch_size):
            pred_np = pred_mask[i].squeeze().cpu().numpy().astype(np.uint8)
            true_np = true_mask[i].squeeze().cpu().numpy().astype(np.uint8)

                     
            if np.sum(true_np) == 0 and np.sum(pred_np) == 0:
                hd95_val = 0.0
            elif np.sum(true_np) == 0 or np.sum(pred_np) == 0:
                hd95_val = 373.0               
            else:
                try:
                    hd95_val = hd95(pred_np, true_np, voxelspacing=self.spacing)
                except:
                    hd95_val = 373.0              

            hd95_vals[i] = hd95_val

        return hd95_vals

    def update(self, dice_vals, iou_vals, hd95_vals):
        """更新指标收集器"""
        et_dice, tc_dice, wt_dice = dice_vals
        et_iou, tc_iou, wt_iou = iou_vals
        et_hd95, tc_hd95, wt_hd95 = hd95_vals            

                    
        self.et_dice.extend(et_dice.cpu().numpy())
        self.tc_dice.extend(tc_dice.cpu().numpy())
        self.wt_dice.extend(wt_dice.cpu().numpy())
        self.et_iou.extend(et_iou.cpu().numpy())
        self.tc_iou.extend(tc_iou.cpu().numpy())
        self.wt_iou.extend(wt_iou.cpu().numpy())

                   
        self.et_hd95.extend(et_hd95.cpu().numpy())
        self.tc_hd95.extend(tc_hd95.cpu().numpy())
        self.wt_hd95.extend(wt_hd95.cpu().numpy())

    def compute_stats(self):
        """计算所有收集样本的统计信息"""

        def _compute(arr):
            if len(arr) == 0:
                return 0.0, 0.0, "0.000 ± 0.000"
            mean = np.mean(arr)
            std = np.std(arr)
            return mean, std, f"{mean:.4f} ± {std:.4f}"

                  
        et_dice_mean, et_dice_std, et_dice_str = _compute(self.et_dice)
        tc_dice_mean, tc_dice_std, tc_dice_str = _compute(self.tc_dice)
        wt_dice_mean, wt_dice_std, wt_dice_str = _compute(self.wt_dice)

                 
        et_iou_mean, et_iou_std, et_iou_str = _compute(self.et_iou)
        tc_iou_mean, tc_iou_std, tc_iou_str = _compute(self.tc_iou)
        wt_iou_mean, wt_iou_std, wt_iou_str = _compute(self.wt_iou)

                     
        et_hd95_mean, et_hd95_std, et_hd95_str = _compute(self.et_hd95)
        tc_hd95_mean, tc_hd95_std, tc_hd95_str = _compute(self.tc_hd95)
        wt_hd95_mean, wt_hd95_std, wt_hd95_str = _compute(self.wt_hd95)

                           
        avg_dice = (et_dice_mean + tc_dice_mean + wt_dice_mean) / 3
        avg_iou = (et_iou_mean + tc_iou_mean + wt_iou_mean) / 3
                     
        avg_hd95 = (et_hd95_mean + tc_hd95_mean + wt_hd95_mean) / 3

        avg_dice_str = f"{avg_dice:.4f} ± {np.std([et_dice_mean, tc_dice_mean, wt_dice_mean]):.4f}"
        avg_iou_str = f"{avg_iou:.4f} ± {np.std([et_iou_mean, tc_iou_mean, wt_iou_mean]):.4f}"
                      
        avg_hd95_str = f"{avg_hd95:.4f} ± {np.std([et_hd95_mean, tc_hd95_mean, wt_hd95_mean]):.4f}"

        return {
            "dice": {
                "mean": avg_dice,
                "std": None,
                "str": avg_dice_str,
                "regions": {
                    "ET": {"mean": et_dice_mean, "std": et_dice_std, "str": et_dice_str},
                    "TC": {"mean": tc_dice_mean, "std": tc_dice_std, "str": tc_dice_str},
                    "WT": {"mean": wt_dice_mean, "std": wt_dice_std, "str": wt_dice_str}
                }
            },
            "iou": {
                "mean": avg_iou,
                "std": None,
                "str": avg_iou_str,
                "regions": {
                    "ET": {"mean": et_iou_mean, "std": et_iou_std, "str": et_iou_str},
                    "TC": {"mean": tc_iou_mean, "std": tc_iou_std, "str": tc_iou_str},
                    "WT": {"mean": wt_iou_mean, "std": wt_iou_std, "str": wt_iou_str}
                }
            },
                       
            "hd95": {
                "mean": avg_hd95,
                "std": None,
                "str": avg_hd95_str,
                "regions": {
                    "ET": {"mean": et_hd95_mean, "std": et_hd95_std, "str": et_hd95_str},
                    "TC": {"mean": tc_hd95_mean, "std": tc_hd95_std, "str": tc_hd95_str},
                    "WT": {"mean": wt_hd95_mean, "std": wt_hd95_std, "str": wt_hd95_str}
                }
            }
        }

if __name__ == "__main__":
                    
    import pyvista as pv

    sphere = pv.Sphere()
    sphere.plot()
    metrics = SegmentationMetrics(num_classes=2, ignore_background=True)

                                   
    pred = torch.randn(2, 2, 100, 100, 100)              
    target = torch.randint(0, 2, (2, 100, 100, 100))        

          
    dice, iou = metrics(pred, target)
    print(f"Dice: {dice:.4f}, IoU: {iou:.4f}")                 