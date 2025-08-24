import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import numpy as np 
from PIL import Image
import random
from data.augmentation_cifar import CIFAR10Policy, CIFAR10v1Policy, RandAugment
class Normalize(nn.Module):
    """
    The class implements the p-norm layer.
    """
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p
    def forward(self, x):
        return F.normalize(x, p=self.p, dim=1)
def crop_from_center(pil_image, new_h, new_w):
    width, height = pil_image.size  # Get dimensions
    left = (width - new_w) / 2
    top = (height - new_h) / 2
    right = (width + new_w) / 2
    bottom = (height + new_h) / 2
    # Crop the center of the image
    pil_image = pil_image.crop((left, top, right, bottom))
    return pil_image
import copy
import warnings
import math
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        self.dinohead = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        for m in self.backbone.parameters():
            m.requires_grad = False
        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in self.backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= args.grad_from_block:
                    m.requires_grad = True
    def forward(self, x):
        x = self.backbone.prepare_tokens(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        x = x[:, 0]
        return self.dinohead(x)
    
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x, puzzle=False):
        prev_feats = x.clone()
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        logits = self.last_layer(x)

        return x_proj, logits

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]
def rgb_jittering(im):
    im = np.array(im, 'int32')
    for ch in range(3):
        im[:, :, ch] += np.random.randint(-2, 2)
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')
class TripletContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""
    def __init__(self, base_transform_weak, base_transform_strong, n_views=2):
        self.base_transform = base_transform_strong
        self.base_transform_weak = base_transform_weak
        self.n_views = n_views
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = 3
        crop_pct = 0.875
        image_size = 224
        self.transform_noform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        self.transform_fixmatch = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        self.transform_fixmatchv2 = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(),
            RandAugment(2, 8),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])


    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x), self.transform_fixmatch(x), self.transform_fixmatch(x), self.transform_fixmatchv2(x), self.transform_fixmatchv2(x)]
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
    def forward(self, features, labels=None, mask=None, args=None, epoch=None, mode=False, all_class=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        weight_newold = torch.ones(batch_size).cuda()
        if epoch>30:
            for i in range(batch_size):
                if labels[i]>=100:
                    weight_newold[i] = 15
        # The above code is not doing anything as it only contains a single word "loss" and is not a
        # valid Python code.
        loss = (loss.view(anchor_count, batch_size)*weight_newold).mean()
        return loss
def info_nce_logits(features, n_views=2, temperature=1.0, device='cuda'):
    b_ = 0.5 * int(features.size(0))
    labels = torch.cat([torch.arange(b_) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    logits = logits / temperature
    return logits, labels
def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
class DistillLoss(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self, student_output, mask_lab, teacher_output, epoch, class_labels, classnum,student_proj, mode):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        out_softlabel = teacher_out
        teacher_out = teacher_out.detach().chunk(2)
        
        class_labels_new = torch.cat([class_labels, class_labels], dim=0)
        loss_all_val = []
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss_all_val.append(loss)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        loss_all_val = torch.cat(loss_all_val)
        ind_sorted = np.argsort(loss_all_val.cpu().data).cuda()
        bs = len(class_labels)
        tmp = out_softlabel[:bs]
        out_softlabel[:bs] = out_softlabel[bs:]
        out_softlabel[bs:] = tmp
        teacher_labels = torch.argmax(out_softlabel, dim=-1)
        #teacher_labels_unlabeled = teacher_labels[torch.cat([~mask_lab, ~mask_lab], dim=0)]
        #student_out_unlabeled = F.softmax(torch.cat([student_out[0], student_out[1]], dim=0)[torch.cat([~mask_lab, ~mask_lab], dim=0)], dim=-1)
        #preds_noisy_flag = F.cross_entropy(student_out_unlabeled, teacher_labels_unlabeled, reduce=False)
        #preds_noisy_flag = preds_noisy_flag.argsort().cpu().numpy()
        #preds_noisy_flag = np.array([1/(preds_noisy_flag[i]*0.07+0.2)*0.8+0.2 for i in range(len(preds_noisy_flag))])
        return total_loss, out_softlabel, ind_sorted#, preds_noisy_flag
class DistillLossv2(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self, student_output, teacher_output, epoch, class_labels, classnum,student_proj, mode,start_epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        out_softlabel = teacher_out
        teacher_out = teacher_out.detach().chunk(2)
        class_labels_new = torch.cat([class_labels, class_labels], dim=0)
        loss_all_val = []
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss_all_val.append(loss)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        loss_all_val = torch.cat(loss_all_val)
        ind_sorted = np.argsort(loss_all_val.cpu().data).cuda()
        loss_all_val = loss_all_val
        bs = len(class_labels_new)
        return total_loss, out_softlabel, ind_sorted
class DistillLossv3(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self, student_output, teacher_output, epoch, class_labels, classnum, unlabeled_labels, mode):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        if mode==1:
            unlabeled_labels = torch.cat([unlabeled_labels, unlabeled_labels], dim=0).cuda()
            teacher_output = teacher_output*0.8 + unlabeled_labels*0.2
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        out_softlabel = teacher_out
        teacher_out = teacher_out.detach().chunk(2)
        class_labels_new = torch.cat([class_labels, class_labels], dim=0)
        loss_all_val = []
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss_all_val.append(loss)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        loss_all_val = torch.cat(loss_all_val)
        ind_sorted = np.argsort(loss_all_val.cpu().data).cuda()
        loss_all_val = loss_all_val
        bs = len(class_labels_new)
        return total_loss, out_softlabel, ind_sorted
class DistillLossv4(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self, student_output, teacher_output, epoch, class_labels, classnum,student_proj, mode,confidences_bs,start_epoch, student_out_weak, args):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        out_softlabel = teacher_out
        teacher_out = teacher_out.detach().chunk(2)
        class_labels_new = torch.cat([class_labels, class_labels], dim=0)
        loss_all_val = []
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                loss_all_val.append(loss)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        loss_all_val = torch.cat(loss_all_val)
        ind_sorted = np.argsort(loss_all_val.cpu().data).cuda()
        loss_all_val = loss_all_val
        bs = len(class_labels_new)
        return total_loss, out_softlabel, ind_sorted
    
class DistillLossv5(nn.Module):
    def __init__(self, warmup_teacher_temp_epochs, nepochs, 
                 ncrops=2, warmup_teacher_temp=0.07, teacher_temp=0.04,
                 student_temp=0.1):
        super().__init__()
        self.student_temp = student_temp
        self.ncrops = ncrops
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_output / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        return total_loss
