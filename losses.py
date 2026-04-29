                    
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import torch.nn.functional as F

class GeneralizedSupervisedNTXenLoss(nn.Module):
    def __init__(self, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.0):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)

        :param temperature:
        :param return_logits:
        """

                                              
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        z_i = func.normalize(z_i, p=2, dim=-1)             
        z_j = func.normalize(z_j, p=2, dim=-1)             
        sim_zii= (z_i @ z_i.T) / self.temperature                                                        
        sim_zjj = (z_j @ z_j.T) / self.temperature                                                        
        sim_zij = (z_i @ z_j.T) / self.temperature                                                                                         
                                                                  
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy()          
        weights = self.kernel(all_labels, all_labels)           
        weights = weights * (1 - np.eye(2*N))                         
        weights /= weights.sum(axis=1)
                                                                                            
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1), torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0)           
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        loss = -1./N * (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum()

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss

    def __str__(self):
        return "{}(temp={}, kernel={}, sigma={})".format(type(self).__name__, self.temperature,
                                                         self.kernel.__name__, self.sigma)



class NTXenLoss(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Constrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, arXiv 2020
    """

    def __init__(self, temperature=0.1, return_logits=False):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8
        self.return_logits = return_logits

    def forward(self, z_i, z_j):
        N = len(z_i)
        z_i = func.normalize(z_i, p=2, dim=-1)             
        z_j = func.normalize(z_j, p=2, dim=-1)             
        sim_zii= (z_i @ z_i.T) / self.temperature                                                        
        sim_zjj = (z_j @ z_j.T) / self.temperature                                                        
        sim_zij = (z_i @ z_j.T) / self.temperature                                                                                         
                                                                  
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)
        correct_pairs = torch.arange(N, device=z_i.device).long()
        loss_i = func.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
        loss_j = func.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)

        if self.return_logits:
            return (loss_i + loss_j), sim_zij, correct_pairs

        return (loss_i + loss_j)

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)


class SupervisedContrastiveLossorginal(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Compute the supervised contrastive loss.

        Args:
        - features: Tensor of shape (batch_size, feature_dim)
        - labels: Tensor of shape (batch_size,)
        - temperature: Temperature parameter

        Returns:
        - loss: Scalar loss value
        """
        batch_size = features.shape[0]
        labels = labels.unsqueeze(-1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

                                   
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / self.temperature

                                                        
        sim_matrix = sim_matrix - torch.eye(batch_size, device=features.device) * 1e9

                                               
        exp_sim_matrix = torch.exp(sim_matrix)
        numerator = torch.sum(exp_sim_matrix * mask, dim=1)
        denominator = torch.sum(exp_sim_matrix, dim=1)

                          
        loss = -torch.log(numerator / denominator).mean()

        return loss

class SupervisedContrastiveLoss_old(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss_old, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j, labels):
        """
        Compute the supervised contrastive loss for multi-modal inputs treated as separate samples.

        Args:
        - z_i: Tensor of shape (batch_size, feature_dim) for the first modality
        - z_j: Tensor of shape (batch_size, feature_dim) for the second modality
        - labels: Tensor of shape (batch_size,)
        - temperature: Temperature parameter

        Returns:
        - loss: Scalar loss value
        """
        batch_size = z_i.shape[0]

                                             
        features = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([labels, labels], dim=0)

                                          
        labels = labels.unsqueeze(-1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

                                   
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / self.temperature

                                                        
        sim_matrix = sim_matrix - torch.eye(2 * batch_size, device=features.device) * 1e9

                                               
        exp_sim_matrix = torch.exp(sim_matrix)
        numerator = torch.sum(exp_sim_matrix * mask, dim=1)
        denominator = torch.sum(exp_sim_matrix, dim=1)

                          
        loss = -torch.log(numerator / denominator).mean()

        return loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, init_temp=0.2, eps=1e-8):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.eps = eps

    def forward(self, z_i, z_j, labels):
                     
        assert z_i.device == z_j.device == labels.device
        batch_size = z_i.shape[0]

                          
        features = torch.cat([z_i, z_j], dim=0)
        features = F.normalize(features, p=2, dim=1)         

                   
        labels_concat = torch.cat([labels, labels], dim=0)

                                                                    
                                    
        indices = torch.arange(batch_size, device=features.device)
        rows = torch.cat([indices, indices + batch_size])                                 
        cols = torch.cat([indices + batch_size, indices])                                 
        same_instance_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=features.device)
        same_instance_mask[rows, cols] = 1.0             

                  
        class_mask = torch.eq(labels_concat.unsqueeze(1), labels_concat.unsqueeze(0)).float()

                              
        mask = (class_mask + same_instance_mask).clamp(max=1.0)         
                                   
                                                                          

                            
        temp = self.temperature.clamp(min=0.05, max=1)
        sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2) / temp

                      
        eye_mask = torch.eye(2 * batch_size, device=features.device)
        sim_matrix = sim_matrix - eye_mask * 1e9

                
        exp_sim = torch.exp(sim_matrix)
        numerator = torch.sum(exp_sim * mask, dim=1)              
        denominator = torch.sum(exp_sim, dim=1) + self.eps              

        loss = -torch.log(numerator / denominator).mean()

        return loss


class DualModalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
                                                                               
        self.temperature = temperature
        self.eps = eps

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: 模态1的特征 [N, D]
            z_j: 模态2的特征 [N, D]
        """
        batch_size = z_i.size(0)
        device = z_i.device

                           
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

                               
                          
        indices = torch.arange(batch_size, device=device)
        pos_indices = torch.stack([indices, indices + batch_size], dim=1)

                             
        features = torch.cat([z_i, z_j], dim=0)           
                                                       
        temp = self.temperature
        sim_matrix = torch.mm(features, features.T) / temp            

                             
        mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
                        
        mask[pos_indices[:, 0], pos_indices[:, 1]] = True
        mask[pos_indices[:, 1], pos_indices[:, 0]] = True

                            
        self_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
        mask = mask & (~self_mask)                    

                            
        exp_sim = torch.exp(sim_matrix)

                 
        pos_sim = torch.sum(exp_sim * mask.float(), dim=1)        

                             
                                                                       
                         
        all_sim = torch.sum(exp_sim * (~self_mask).float(), dim=1)        

              
        losses = -torch.log(pos_sim / (all_sim + self.eps))
        loss = losses.mean()

        return loss


                                                
                                                                                
                                                                    
                            
                                                                    
                                                    
                                        
                                                        
                                                    
                         
 
                                               
                      
                                          
 
                   
                                                                          
 
                            
                                                  
                                                              
 
                                    
                           
                                                     
 
                     
                                     
                             
                                                             
                           
                                                
               
 
                     
                                    
                             
                                                             
                           
                                                
                      
 
                               
                                                                      
 
                        
                                                                          
 
                                    
                                                                  
                              
 
                                          
                                        
                                  
 
                  
                                                              
 
                   
                                                                        
                                          
                                                   
 
                   
                                                                                
 
                         
                                           
                                                                         
 
                 
                                                         
                                                                                    
                                                           
 
                                                             
                                                         
 
                  
                                                         
 
                              
                                                    
 
                   
                                                                         
 
                       
                                                                                      
                                                                                 
                       
                                                                             
                                        
                         
                                                           
                                                                                      
                                                       
 
                                                     
                                                                     
 
                                                            
                                                                                       
                                                   
                                                         
                                                                         
 
                                     
                                                                              
 
                                     
                                                                              
 
                        
                                                              
                                      
                                      
 
                          
                                                       
                                                                     
 
                          
                                                                                        
                                                    
 
                                                         
 
                         
                                                                        
                                                                                                            
 
                  
                                                                       
 
                  
                                                                        
                                                  
 
                
                                         
                                                            
                                                                                     
 
                                                                        
                     

class BoundaryAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, boundary_dilation=2, max_samples=1000,
                 use_region_contrast=True):
        super().__init__()
                                                                    
        self.temperature = temperature
        self.boundary_dilation = boundary_dilation
        self.max_samples = max_samples
        self.use_region_contrast = use_region_contrast
        self.eps = 1e-8

                                               
                      
                                          
     
                   
                                                                          
     
                           
                                                  
                                                              
     
                    
                                                    
     
                            
                               
                           
                                     
                                  
                                        
                           
                           
               
     
                             
                                    
                                  
                                        
                           
                           
                             
     
                    
                                                                
     
                    
                                                              
     
                         
                                                                                               
     
                         

    def _generate_boundary_mask(self, label):
                    
        label_int = label.to(torch.long)

                 
        boundary_mask = torch.zeros_like(label_int, dtype=torch.float32)

                          
        unique_classes = torch.unique(label_int)
        unique_classes = unique_classes[unique_classes != 0]

        for cls in unique_classes:
                         
            class_mask = (label_int == cls).float()

                   
            dilated = F.max_pool3d(
                class_mask,
                kernel_size=2 * self.boundary_dilation + 1,
                stride=1,
                padding=self.boundary_dilation
            )

                   
            eroded = F.avg_pool3d(
                class_mask,
                kernel_size=2 * self.boundary_dilation + 1,
                stride=1,
                padding=self.boundary_dilation
            ) > 0.99

                             
            class_boundary = (dilated - eroded.float()).clamp(min=0)

                      
            boundary_mask = torch.maximum(boundary_mask, class_boundary)

                                  
        boundary_mask = boundary_mask * (label_int != 0).float()
        return boundary_mask

    def forward(self, features, labels):
        B, C, D, H, W = features.shape
        device = features.device

                
        boundary_mask = self._generate_boundary_mask(labels)
        boundary_confidence = boundary_mask               

                 
        flat_features = features.permute(0, 2, 3, 4, 1).reshape(-1, C)
        flat_labels = labels.view(-1, 1)
        flat_boundary = boundary_mask.view(-1, 1)
        flat_confidence = boundary_confidence.view(-1)

                 
        boundary_indices = (flat_boundary[:, 0] > 0).nonzero(as_tuple=True)[0]

                       
        if boundary_indices.numel() == 0:
            print("Warning: No boundary points detected. Loss set to 0.")
            return torch.tensor(0.0, device=device, requires_grad=True)

               
        if boundary_indices.size(0) > self.max_samples:
            rand_idx = torch.randperm(boundary_indices.size(0))[:self.max_samples]
            boundary_indices = boundary_indices[rand_idx]

        boundary_features = flat_features[boundary_indices]
        boundary_labels = flat_labels[boundary_indices]
        boundary_confidence = flat_confidence[boundary_indices]          

                
        unique_classes = torch.unique(labels)
        num_classes = unique_classes.numel() - (0 in unique_classes)         

                
                                                          
        temp = self.temperature

                        
        if num_classes == 1 and self.use_region_contrast:
                     
            bg_indices = (flat_labels[:, 0] == 0).nonzero(as_tuple=True)[0]
                           
            fg_indices = ((flat_labels[:, 0] > 0) & (flat_boundary[:, 0] < 0.5)).nonzero(as_tuple=True)[0]

                            
            if bg_indices.numel() == 0 or fg_indices.numel() == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)

                   
            if fg_indices.size(0) > self.max_samples:
                rand_idx = torch.randperm(fg_indices.size(0))[:self.max_samples]
                fg_indices = fg_indices[rand_idx]
            fg_features = flat_features[fg_indices]

                   
            bg_sample_size = min(self.max_samples, bg_indices.size(0))
            rand_idx = torch.randperm(bg_indices.size(0))[:bg_sample_size]
            bg_indices = bg_indices[rand_idx]
            bg_features = flat_features[bg_indices]

                   
            boundary_features_norm = F.normalize(boundary_features, p=2, dim=1)
            fg_features_norm = F.normalize(fg_features, p=2, dim=1)
            bg_features_norm = F.normalize(bg_features, p=2, dim=1)

                                
            pos_sim = torch.mm(boundary_features_norm, fg_features_norm.t())

                                
            neg_sim = torch.mm(boundary_features_norm, bg_features_norm.t())

                                
            self_sim = torch.mm(boundary_features_norm, boundary_features_norm.t())
            diag_mask = 1 - torch.eye(boundary_features.size(0), device=device)
            intra_sim = self_sim * diag_mask        

                  
            pos_sim = pos_sim / temp
            neg_sim = neg_sim / temp
            intra_sim = intra_sim / temp

                    
            pos_exp = torch.exp(pos_sim).sum(dim=1)
            intra_exp = torch.exp(intra_sim).sum(dim=1)
            neg_exp = torch.exp(neg_sim).sum(dim=1)

            numerator = pos_exp + intra_exp
            denominator = numerator + neg_exp

                    
            ratio = torch.clamp(numerator / (denominator + self.eps), min=1e-12, max=1.0)
            region_loss = -torch.log(ratio)

                       
            region_loss = (region_loss * boundary_confidence).mean()

            return region_loss

                        
        else:
                   
            boundary_features_norm = F.normalize(boundary_features, p=2, dim=1)
            sim_matrix = torch.mm(boundary_features_norm, boundary_features_norm.t())

                    
            sim_matrix = sim_matrix / temp

                                
            label_mask = (boundary_labels == boundary_labels.t()).float()

                    
            eye_mask = torch.eye(boundary_features.size(0), device=device)
            label_mask = label_mask * (1 - eye_mask)

                      
            valid_pairs = label_mask.sum(1) > 0
            if not valid_pairs.any():
                return torch.tensor(0.0, device=device, requires_grad=True)

                  
            exp_sim = torch.exp(sim_matrix)
            numerator = torch.sum(exp_sim * label_mask, dim=1)
            denominator = torch.sum(exp_sim, dim=1) - torch.exp(torch.diag(sim_matrix))

                    
            valid_loss_mask = valid_pairs.float()
            losses = -torch.log(numerator / (denominator + self.eps))
            weighted_losses = losses * boundary_confidence * valid_loss_mask

                  
            loss = weighted_losses.sum() / (valid_loss_mask.sum() + self.eps)
            return loss

class DeformationRegularizationLoss(nn.Module):
    """
    正则化位移场（displacement field）以保证平滑性和避免折叠。
    输入: List of displacement fields [B, 3, D, H, W]
    输出: 标量正则化损失
    """
    def __init__(self, smooth_weight=0.1, jacobian_weight=0.1, eps=1e-8):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.jacobian_weight = jacobian_weight
        self.eps = eps

    def _compute_smoothness_loss(self, disp):
        """计算位移场的二阶梯度平滑损失"""
                
        dx = (disp[:, :, 1:, :, :] - disp[:, :, :-1, :, :]).pow(2).mean()
        dy = (disp[:, :, :, 1:, :] - disp[:, :, :, :-1, :]).pow(2).mean()
        dz = (disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]).pow(2).mean()
        return dx + dy + dz

    def _compute_jacobian_det_loss(self, disp):
        """
        计算雅可比行列式的负值惩罚（鼓励 det(J) > 0）
        disp: [B, 3, D, H, W] —— 注意通道顺序为 (x, y, z) 对应 (W, H, D)
        """
        B, C, D, H, W = disp.shape
        assert C == 3, "Displacement field must have 3 channels"

                                    
                          
                                                  
        u = disp[:, 0, :, :, :]                      
        v = disp[:, 1, :, :, :]                      
        w = disp[:, 2, :, :, :]                      

               
        du_dx = torch.zeros_like(u)
        du_dx[:, :, :, :-1] = u[:, :, :, 1:] - u[:, :, :, :-1]
        du_dx[:, :, :, -1] = du_dx[:, :, :, -2]

               
        du_dy = torch.zeros_like(u)
        du_dy[:, :, :-1, :] = u[:, :, 1:, :] - u[:, :, :-1, :]
        du_dy[:, :, -1, :] = du_dy[:, :, -2, :]

               
        du_dz = torch.zeros_like(u)
        du_dz[:, :-1, :, :] = u[:, 1:, :, :] - u[:, :-1, :, :]
        du_dz[:, -1, :, :] = du_dz[:, -2, :, :]

               
        dv_dx = torch.zeros_like(v)
        dv_dx[:, :, :, :-1] = v[:, :, :, 1:] - v[:, :, :, :-1]
        dv_dx[:, :, :, -1] = dv_dx[:, :, :, -2]

               
        dv_dy = torch.zeros_like(v)
        dv_dy[:, :, :-1, :] = v[:, :, 1:, :] - v[:, :, :-1, :]
        dv_dy[:, :, -1, :] = dv_dy[:, :, -2, :]

               
        dv_dz = torch.zeros_like(v)
        dv_dz[:, :-1, :, :] = v[:, 1:, :, :] - v[:, :-1, :, :]
        dv_dz[:, -1, :, :] = dv_dz[:, -2, :, :]

               
        dw_dx = torch.zeros_like(w)
        dw_dx[:, :, :, :-1] = w[:, :, :, 1:] - w[:, :, :, :-1]
        dw_dx[:, :, :, -1] = dw_dx[:, :, :, -2]

               
        dw_dy = torch.zeros_like(w)
        dw_dy[:, :, :-1, :] = w[:, :, 1:, :] - w[:, :, :-1, :]
        dw_dy[:, :, -1, :] = dw_dy[:, :, -2, :]

               
        dw_dz = torch.zeros_like(w)
        dw_dz[:, :-1, :, :] = w[:, 1:, :, :] - w[:, :-1, :, :]
        dw_dz[:, -1, :, :] = dw_dz[:, -2, :, :]

                             
                                                           
                                        
        det_J = (
            (1 + du_dx) * (1 + dv_dy) * (1 + dw_dz)
            + du_dy * dv_dz * dw_dx
            + du_dz * dv_dx * dw_dy
            - (1 + du_dx) * dv_dz * dw_dy
            - (1 + dv_dy) * du_dz * dw_dx
            - (1 + dw_dz) * du_dy * dv_dx
        )

                                
        folding_loss = torch.clamp(-det_J, min=0).mean()
        return folding_loss

    def forward(self, deform_params_list):
        """
        deform_params_list: List of [B, 3, D, H, W] tensors from alignment units
        """
        if not deform_params_list:
            return torch.tensor(0.0, device=deform_params_list[0].device if deform_params_list else 'cpu', requires_grad=True)

        total_loss = 0.0
        for disp in deform_params_list:
            smooth_loss = self._compute_smoothness_loss(disp)
            jacobian_loss = self._compute_jacobian_det_loss(disp)
            total_loss += self.smooth_weight * smooth_loss + self.jacobian_weight * jacobian_loss

        return total_loss / len(deform_params_list)


class CombinedGlobalLocalLoss(nn.Module):
    def __init__(self, global_temp=0.1, local_temp=0.1, boundary_dilation=2,                   
                 max_samples=1000, global_weight=0.5, local_weight=0.5):
        super().__init__()
        self.global_loss = DualModalContrastiveLoss(temperature=global_temp)
        self.local_loss = BoundaryAwareContrastiveLoss(
            temperature=local_temp,
            boundary_dilation=boundary_dilation,
            max_samples=max_samples
        )
        self.reg_loss_fn = DeformationRegularizationLoss()          
        self.global_weight = global_weight
        self.local_weight = local_weight

                
    @property
    def temperature(self):
        """返回所有温度参数的列表（用于优化器分组）"""
        return [self.global_loss.temperature, self.local_loss.temperature]

    def forward(self, z_global_i, z_global_j, z_local_i, z_local_j, labels_i, labels_j, deform_params_list):
        """
        计算全局和局部联合对比损失

        参数:
            z_global_i, z_global_j: 两个模态的全局特征 [B, D]
            z_local_i, z_local_j: 两个模态的局部特征图 [B, C, D, H, W]
            labels_i, labels_j: 两个模态的分割标签 [B, 1, D, H, W]
        """
                
        loss_global = self.global_loss(z_global_i, z_global_j)

                    
        loss_local_i = self.local_loss(z_local_i, labels_i)
        loss_local_j = self.local_loss(z_local_j, labels_j)
        loss_local = (loss_local_i + loss_local_j) / 2

        if deform_params_list is not None:
            reg_loss = self.reg_loss_fn(deform_params_list)

              
        total_loss = self.global_weight * loss_global + self.local_weight * loss_local + reg_loss

        return total_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, epsilon=1e-6, bg_weight=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
                                  
        self.weights = torch.ones(num_classes)
        self.weights[0] = bg_weight            

    def forward(self, pred, target):
                                           
                                            

                  
        pred = F.softmax(pred, dim=1)

                                            
        target_onehot = F.one_hot(target.long(), self.num_classes).permute(0, 4, 1, 2, 3).float()

                     
        weights = self.weights.to(pred.device)

                      
        intersection = torch.sum(pred * target_onehot, dim=(2, 3, 4))          
        union = torch.sum(pred, dim=(2, 3, 4)) + torch.sum(target_onehot, dim=(2, 3, 4))          

                       
        dice_per_class = (2. * intersection + self.epsilon) / (union + self.epsilon)          

                                  
        dice_loss_per_class = 1 - dice_per_class          

                 
        weighted_loss = dice_loss_per_class * weights          

                               
        return weighted_loss.mean(dim=0).sum() / weights.sum()

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss