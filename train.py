import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse
import math
import numpy as np
import torch
import torch.nn as nn 
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits
from util.general_utils import AverageMeter, init_experiment
from util.cluster_and_log_utils import log_accs_from_preds, log_accs_from_preds_v2, split_cluster_acc_v2_label
from config import exp_root
from model import Model, DINOHead, info_nce_logits, SupConLoss, DistillLossv2, TripletContrastiveLearningViewGenerator, ContrastiveLearningViewGenerator, get_params_groups
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import random
from collections import Counter
@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
class SimCLR(nn.Module):
    def __init__(self, temp=0.2):
        super().__init__()
        
        self.temp = temp
        
    def contrastive_loss(self, q, k):
        
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        
        # gather all targets
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temp
        N = logits.shape[0] 
        
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.temp)
    def forward(self, student_output, teacher_output):
        student_out = student_output
        student_out = student_out.chunk(2)
        teacher_out = teacher_output 
        teacher_out = teacher_out.detach().chunk(2)
        return self.contrastive_loss(student_out[0], teacher_out[1]) + self.contrastive_loss(student_out[1], teacher_out[0])
def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return 20*float(current)
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        outputs_u = outputs_u/0.7
        outputs_x = outputs_x/0.7
        outputs_u = outputs_u / outputs_u.sum(dim=1, keepdim=True) 
        outputs_x = outputs_x / outputs_x.sum(dim=1, keepdim=True) 
        targets_x = targets_x / targets_x.sum(dim=1, keepdim=True) 
        targets_u = targets_u / targets_u.sum(dim=1, keepdim=True) 
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((outputs_u - targets_u)**2)
        return Lx, Lu, linear_rampup(epoch,warm_up)
def entropy(x):
    EPS = 1e-8
    x_ =  torch.clamp(x, min = EPS)
    b =  x_ * torch.log(x_)
    if len(b.size()) == 2: # Sample-wise entropy
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))
class MarginLoss(nn.Module):
    
    def __init__(self, m=0.2, weight=None, s=10):
        super(MarginLoss, self).__init__()
        self.m = m
        self.s = s
        self.weight = weight
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        x_m = x - self.m * self.s
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(output, target, weight=self.weight)
class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
import codecs
import csv
def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name,'w+','utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)
    print("Save to csv file. Finished.")
def train(student, train_loader, train_loader_all, test_loader, unlabelled_train_loader, args, classnum,train_noformdataset_len, train_loader_all_warmup):
    params_groups = get_params_groups(student)
    start_epoch = args.start_epoch
    flag_mixup = args.flag_mixup
    optimizer = SGD(params_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    fp16_scaler = None
    if args.fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 1e-3,
        )
    cluster_criterion = DistillLossv2(
                        args.warmup_teacher_temp_epochs,
                        args.epochs,
                        args.n_views,
                        args.warmup_teacher_temp,
                        args.teacher_temp,
                    )
    semi_criterion = SemiLoss()
    # inductive
    best_test_acc_ubl = 0
    # transductive
    best_train_acc_lab = 0
    best_train_acc_ubl = 0 
    best_train_acc_all = 0
    
    best_train_acc_lab_new = 0
    best_train_acc_ubl_new = 0 
    best_train_acc_all_new = 0
    
    device = torch.device('cuda:0')
    lens = len(train_loader.dataset)
    num_iter = (len(train_loader.dataset)//args.batch_size)+1
    len_dataset = train_loader.dataset.__len__()
    history_queue_predict_weak = [[] for i in range(len_dataset)]
    history_queue_predict_strong = [[] for i in range(len_dataset)]    
    history_queue_predict_single = [[] for i in range(len_dataset)]   
    lenlabeled = train_loader.dataset.__lenlabel__()
    lenunlabeled = train_loader.dataset.__lenunlabel__()
    history_labels_onehot_all_weak, history_labels_onehot_all_strong = torch.zeros(lenunlabeled+lenlabeled) - torch.ones(lenunlabeled+lenlabeled), torch.zeros(lenunlabeled+lenlabeled) - torch.ones(lenunlabeled+lenlabeled) 
    history_labels_onehot_all_weak, history_labels_onehot_all_strong = history_labels_onehot_all_weak.long(), history_labels_onehot_all_strong.long()

    total_acc_label_all, old_acc_label_all, new_acc_label_all = [], [], []


    flag_semi = False
    print('Start Semi-supervised Training!')
    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        temp_teacher = cluster_criterion.teacher_temp_schedule[epoch]
        student = student.to(device)
        max_queue = 17
        bar_line_fix = 13+epoch/args.epochs*2       
        student.train()
        threshold_perclass = torch.zeros(classnum)
        allclass_queue = []
        allclass_queue_modenum = []
        for p in range(len(history_queue_predict_weak)):
            if len(history_queue_predict_weak[p]) == 0:
                continue
            queue_cur = torch.cat(history_queue_predict_weak[p], dim=0).reshape(-1, classnum)
            counts = np.bincount(queue_cur.argmax(dim=-1).cpu())
            mode_num, maxnum_times= np.argmax(counts), counts.max()
            if threshold_perclass[mode_num]<maxnum_times:
                threshold_perclass[mode_num] = maxnum_times
        total_acc_label, old_acc_label, new_acc_label = 0, 0, 0
        for batch_idx, batch in enumerate(train_loader):
            images, class_labels, uq_idxs, items, mask_lab = batch
            mask_lab = mask_lab[:, 0]
            class_labels, mask_lab = class_labels.cuda(non_blocking=True), mask_lab.cuda(non_blocking=True).bool()
            images_strong = torch.cat([images[2], images[3]], dim=0).cuda(non_blocking=True)
            images = torch.cat(images[:2], dim=0).cuda(non_blocking=True)
            confidences_bs = torch.ones(len(class_labels)*2).cuda()
            with torch.cuda.amp.autocast(fp16_scaler is not None):
                student_proj, student_out = student(images)
                student_proj, student_out = student_proj[:len(class_labels)*2], student_out[:len(class_labels)*2]
                teacher_out = student_out.detach()
                
                # clustering, unsup
                class_labels_ori = class_labels.clone()
                class_labels = class_labels.detach().cpu()
                mask_lab = mask_lab.detach().cpu()
                mask_lab_sup, mask_lab_semi, mask_lab_unsup = torch.zeros(mask_lab.shape).cuda().long(), torch.zeros(mask_lab.shape).cuda().long(), torch.zeros(mask_lab.shape).cuda().long()
                targets_unlabeled = torch.zeros((len(mask_lab), classnum)).cuda()
                mask_lab_semi_new = torch.zeros(mask_lab.shape).cuda().long()
                mask_lab_semi_old = torch.zeros(mask_lab.shape).cuda().long()
                mask_lab_sup_new = torch.zeros(mask_lab.shape).cuda().long()
                mask_lab_sup_old = torch.zeros(mask_lab.shape).cuda().long()
                mask_lab_sup = mask_lab.clone()

                mask_lab_sup_newv2 = mask_lab.clone()
                if epoch>=20:
                    for p in range(len(student_out)//2):
                         
                        if mask_lab[p]==1:
                            mask_lab_sup_newv2[p] = 1
                            mask_lab_sup[p] = 1
                            continue
                        if len(history_queue_predict_weak[items[p]])<=1:
                            mask_lab_unsup[p] = 1
                            continue
                        queue_cur = torch.cat(history_queue_predict_weak[items[p]], dim=0).reshape(-1, classnum)
                        counts = np.bincount(queue_cur.argmax(dim=-1).cpu())
                        mode_num, maxnum_times= np.argmax(counts), counts.max()
                        queue_cur_strong = torch.cat(history_queue_predict_strong[items[p]], dim=0).reshape(-1, classnum)
                        counts_strong = np.bincount(queue_cur_strong.argmax(dim=-1).cpu())
                        mode_num_strong, maxnum_times_strong= np.argmax(counts_strong), counts.max()
                        if True:
                            if True:
                                if maxnum_times>=bar_line_fix:
                                    mask_lab_sup[p] = 1
                                    class_labels[p] = mode_num
                                    mask_lab_semi[p] = 1
                                    tmp = queue_cur_strong.mean(dim=0).cuda()
                                    targets_unlabeled[p] = tmp / tmp.sum(dim=0, keepdim=True) 
                                else:
                                    if args.mode==1:
                                        if maxnum_times>=bar_line_fix-2 and mode_num_strong==mode_num:
                                            mask_lab_semi[p] = 1
                                            class_labels[p] = mode_num
                                            tmp = queue_cur_strong.mean(dim=0).cuda()
                                            targets_unlabeled[p] = tmp / tmp.sum(dim=0, keepdim=True) 
                                        elif maxnum_times_strong>=bar_line_fix-2 and mode_num_strong==mode_num:
                                            mask_lab_semi[p] = 1
                                            class_labels[p] = mode_num_strong
                                            tmp = queue_cur_strong.mean(dim=0).cuda()
                                            targets_unlabeled[p] = tmp / tmp.sum(dim=0, keepdim=True) 
                                        else:
                                            mask_lab_unsup[p] = 1

                mask_lab_sup_new_ori = mask_lab_sup.clone()
                mask_lab_semi_new_ori = mask_lab_semi.clone()
                if True:
                    unlabeled_images = images_strong
                    _, logits_strong_final = student(unlabeled_images.cuda())
                    if torch.sum(mask_lab_semi)>=1:
                        loss_mixup = 0
                        if True:
                            if torch.sum(mask_lab_sup_new)>=1:
                                mask_lab_sup_new, mask_lab_semi_new = mask_lab_sup_new.bool(), mask_lab_sup_new.bool() 
                                mask_lab_semi_new_cat = torch.cat([mask_lab_semi_new, mask_lab_semi_new], dim=0).bool()
                                mask_lab_sup_new_cat = torch.cat([mask_lab_sup_new, mask_lab_sup_new], dim=0).bool()

                                images_label, images_unlabel = images[mask_lab_sup_new_cat], images[mask_lab_semi_new_cat]

                                class_labels_label, class_labels_unlabel = class_labels.cuda(), targets_unlabeled.cuda()
                                class_labels_label = F.one_hot(class_labels_label, num_classes=class_labels_unlabel.shape[-1])

                                class_labels_label = torch.cat([class_labels_label, class_labels_label], dim=0)[mask_lab_sup_new_cat]
                                class_labels_unlabel = torch.cat([class_labels_unlabel, class_labels_unlabel], dim=0)[mask_lab_semi_new_cat]
                                lens_min = min(len(images_label), len(images_unlabel))
                                images_labelw, images_unlabelw = images_label[:lens_min], images_unlabel[:lens_min]
                                class_labels_label, class_labels_unlabel = class_labels_label[:lens_min].cuda(), class_labels_unlabel[:lens_min].cuda()
                                class_labels_unlabel = class_labels_unlabel / class_labels_unlabel.sum(dim=1, keepdim=True)
                                
                                if args.type==0:
                                    current_class_labels, current_class_unlabels = student_out[mask_lab_sup_new_cat].detach().clone(), student_out[mask_lab_semi_new_cat].detach().clone()
                                    
                                    current_class_labels = (current_class_labels.chunk(2)[0] + current_class_labels.chunk(2)[1])/2.0
                                    current_class_unlabels = (current_class_unlabels.chunk(2)[0] + current_class_unlabels.chunk(2)[1])/2.0

                                    current_class_labels = torch.cat([current_class_labels, current_class_labels], dim=0)[:lens_min]
                                    current_class_unlabels = torch.cat([current_class_unlabels, current_class_unlabels], dim=0)[:lens_min]
                                    
                                    current_class_labels = current_class_labels / current_class_labels.sum(dim=1, keepdim=True)
                                    current_class_unlabels = current_class_unlabels / current_class_unlabels.sum(dim=1, keepdim=True)

                                    current_class_labels = F.softmax(current_class_labels, dim=-1)
                                    current_class_unlabels = F.softmax(current_class_unlabels, dim=-1)

                                    class_labels_label = class_labels_label*0.8 + current_class_labels.cuda()*0.2
                                    class_labels_unlabel = class_labels_unlabel*0.5 + current_class_unlabels.cuda()*0.5
                                    
                                    class_labels_label = class_labels_label**(1/0.7)
                                    class_labels_unlabel = class_labels_unlabel**(1/0.7)
                                
                                all_inputs = torch.cat([images_labelw, images_unlabelw], dim=0).cuda()
                                all_targets = torch.cat([class_labels_label, class_labels_unlabel], dim=0).cuda()
                                idx = torch.randperm(len(all_inputs))
                                input_a, input_b = all_inputs, all_inputs[idx]
                                target_a, target_b = all_targets, all_targets[idx]
                                l = np.random.beta(4, 4)        
                                l = max(l, 1-l)

                                mixed_input = l * input_a + (1 - l) * input_b        
                                mixed_target = l * target_a + (1 - l) * target_b

                                _, logits = student(mixed_input)
                                logits_x = logits[:len(class_labels_label)]
                                logits_u = logits[len(class_labels_label):]    
                                logits_x = logits_x/logits_x.sum(dim=1, keepdim=True)
                                logits_u = logits_u/logits_u.sum(dim=1, keepdim=True)
                                mixed_target = mixed_target / mixed_target.sum(dim=1, keepdim=True)
                                Lx, Lu, lamb = semi_criterion(logits_x, mixed_target[:len(class_labels_label)], logits_u, mixed_target[len(class_labels_label):], epoch+batch_idx/num_iter, 10)
                                loss_mixup = Lx+ Lu*5
                            student_out_tmp = student_out / student_out.sum(dim=1, keepdim=True)
                            loss_mixup += nn.CrossEntropyLoss()(F.softmax(student_out_tmp[:len(student_out)//2]), class_labels.cuda())*1e-10
                    else:
                        student_out_tmp = student_out / student_out.sum(dim=1, keepdim=True)
                        loss_mixup = nn.CrossEntropyLoss()(F.softmax(student_out_tmp[:len(student_out)//2]), class_labels.cuda())*1e-10
                history_labels_onehot_all_weak[items[mask_lab_sup.cpu()==1]] = class_labels.cpu()[mask_lab_sup.cpu()==1]
                class_labels = class_labels.cuda(non_blocking=True)
                mask_lab_sup = mask_lab_sup.cuda(non_blocking=True)
                cluster_loss, out_softlabel, ind_sorted = cluster_criterion(student_out, teacher_out, epoch, class_labels,classnum,student_proj, args.mode, confidences_bs)
                
                # clustering, sup
                sup_logits = torch.cat([f[mask_lab] for f in (student_out / 0.1).chunk(2)], dim=0)
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0).cuda()
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)
                
                teacher_out_mean = out_softlabel.detach().chunk(2)
                teacher_out_mean_weak = F.softmax(teacher_out_mean[0]*0.3+teacher_out_mean[1]*0.3+student_out.detach().chunk(2)[0]*0.3+student_out.detach().chunk(2)[1]*0.3, dim=-1).cpu()
                teacher_out_mean_strong = F.softmax(logits_strong_final.chunk(2)[0].detach().cpu()+logits_strong_final.chunk(2)[1].detach().cpu(), dim=-1)
                
                for p in range(len(teacher_out_mean[0])):
                    history_queue_predict_weak[items[p]].append(teacher_out_mean_weak[p])
                    history_queue_predict_strong[items[p]].append(teacher_out_mean_strong[p])
                    if len(history_queue_predict_weak[items[p]])>max_queue:
                        history_queue_predict_weak[items[p]].pop(0)
                        history_queue_predict_strong[items[p]].pop(0)
                avg_probs = (student_out / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss += args.memax_weight * me_max_loss
                # represent learning, unsup
                contrastive_logits, contrastive_labels = info_nce_logits(features=student_proj)
                contrastive_loss = torch.nn.CrossEntropyLoss()(contrastive_logits, contrastive_labels)
                # representation learning, sup
                student_proj = torch.cat([f[mask_lab_sup_newv2].unsqueeze(1) for f in student_proj.chunk(2)], dim=1)
                student_proj = torch.nn.functional.normalize(student_proj, dim=-1)
                sup_con_labels = class_labels[mask_lab_sup_newv2]
                sup_con_loss = SupConLoss()(student_proj, labels=sup_con_labels, args=args, epoch=epoch)
                pstr = ''
                pstr += f'cls_loss: {cls_loss.item():.4f} '
                pstr += f'cluster_loss: {cluster_loss.item():.4f} '
                pstr += f'sup_con_loss: {sup_con_loss.item():.4f} '
                pstr += f'contrastive_loss: {contrastive_loss.item():.4f} '
                loss = 0
                loss += (1 - args.sup_weight) * cluster_loss + args.sup_weight * cls_loss
                loss += (1 - args.sup_weight) * contrastive_loss + args.sup_weight * sup_con_loss
                loss_mixup = loss_mixup*0.1
                pstr += f'mixuploss: {loss_mixup.item():.4f} ' 
                loss += loss_mixup

            # Train acc
            loss_record.update(loss.item(), class_labels.size(0))
            optimizer.zero_grad()
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
            if batch_idx % args.print_freq == 0:
                args.logger.info('Epoch: [{}][{}/{}]\t loss {:.5f}\t {}'
                            .format(epoch, batch_idx, len(train_loader), loss.item(), pstr))
        

        args.logger.info('Train Epoch: {} Avg Loss: {:.4f} '.format(epoch, loss_record.avg))
        args.logger.info('Testing on unlabelled examples in the training data...')
        all_acc, old_acc, new_acc = test(student, unlabelled_train_loader, epoch=epoch, save_name='Train ACC Unlabelled', args=args)
        args.logger.info('Testing on disjoint test set...')
        all_acc_test, old_acc_test, new_acc_test = test(student, test_loader, epoch=epoch, save_name='Test ACC', args=args)

        args.logger.info('Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
        args.logger.info('Test Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc_test, old_acc_test, new_acc_test))
        # Step schedule
        exp_lr_scheduler.step()
        save_dict = {
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        torch.save(save_dict, args.model_path)
        args.logger.info("model saved to {}.".format(args.model_path))
        if all_acc_test > best_train_acc_all:
            args.logger.info(f'Best ACC on old Classes on disjoint test set: {old_acc_test:.4f}...')
            args.logger.info('Best Train Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, old_acc, new_acc))
            torch.save(save_dict, args.model_path[:-3] + f'_best.pt')
            args.logger.info("model saved to {}.".format(args.model_path[:-3] + f'_best.pt'))
            # inductive
            best_test_acc_ubl = new_acc_test
            # transductive            
            best_train_acc_lab = old_acc_test
            best_train_acc_ubl = new_acc_test
            best_train_acc_all = all_acc_test
        if new_acc_test > best_train_acc_ubl_new:
            best_train_acc_lab_new = old_acc_test
            best_train_acc_ubl_new = new_acc_test
            best_train_acc_all_new = all_acc_test
        args.logger.info(f'Exp Name: {args.exp_name}')
        args.logger.info(f'Metrics with best model on test new set: All: {best_train_acc_all_new:.4f} Old: {best_train_acc_lab_new:.4f} New: {best_train_acc_ubl_new:.4f}')
        args.logger.info(f'Metrics with best model on test set: All: {best_train_acc_all:.4f} Old: {best_train_acc_lab:.4f} New: {best_train_acc_ubl:.4f}')
def test(model, test_loader, epoch, save_name, args):
    model.eval()
    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, logits = model(images)
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes)) else False for x in label]))
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    args=args)
    return all_acc, old_acc, new_acc
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v2', 'v2p'])
    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, fgvc_aricraft, herbarium_19')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', action='store_true', default=True)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--sup_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    
    parser.add_argument('--memax_weight', type=float, default=2)
    parser.add_argument('--warmup_teacher_temp', default=0.07, type=float, help='Initial value for the teacher temperature.')
    parser.add_argument('--teacher_temp', default=0.04, type=float, help='Final value (after linear warmup)of the teacher temperature.')
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int, help='Number of warmup epochs for the teacher temperature.')
    parser.add_argument('--fp16', action='store_true', default=False)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--exp_name', default=None, type=str)
    parser.add_argument('--start_epoch', default=10, type=int)
    parser.add_argument('--flag_mixup', default=False, type=bool)
    parser.add_argument('--mode', default=0, type=int)
    parser.add_argument('--type', default=0, type=int)
    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)
    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    init_experiment(args, runner_name=['simgcd'])
    args.logger.info(f'Using evaluation function {args.eval_funcs[0]} to print results')
    
    torch.backends.cudnn.benchmark = True
    # ----------------------
    # BASE MODEL
    # ----------------------
    args.interpolation = 3
    args.crop_pct = 0.875
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
 
    if args.warmup_model_dir is not None:
        args.logger.info(f'Loading weights from {args.warmup_model_dir}')
        backbone.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))
    
    # NOTE: Hardcoded image size as we do not finetune the entire ViT model
    args.image_size = 224
    args.feat_dim = 768
    args.num_mlp_layers = 3
    args.mlp_out_dim = args.num_labeled_classes + args.num_unlabeled_classes
    # ----------------------
    # HOW MUCH OF BASE MODEL TO FINETUNE
    # ----------------------
    for m in backbone.parameters():
        m.requires_grad = False
    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= args.grad_from_block:
                m.requires_grad = True
    
    args.logger.info('model build')
    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, train_transform_noform, test_transform, train_transform_weak, _ = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = TripletContrastiveLearningViewGenerator(base_transform_weak=train_transform_weak, base_transform_strong = train_transform, n_views=args.n_views)
    # --------------------
    # DATASETS
    # --------------------
    train_dataset, train_noformdataset, test_dataset, unlabelled_train_examples_test, datasets = get_datasets(args.dataset_name,
                                                                                         train_transform, train_transform_noform, 
                                                                                         test_transform,
                                                                                         args)
    # --------------------
    # SAMPLER
    # Sampler which balances labelled and unlabelled examples in each batch
    # --------------------
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))
    # --------------------
    # DATALOADERS
    # --------------------
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
                              sampler=sampler, drop_last=True, pin_memory=True)
    
    train_loader_all = DataLoader(train_noformdataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,drop_last=False, pin_memory=True)
    test_loader_unlabelled = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=256, shuffle=False, pin_memory=False)
    test_loader_labelled = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=256, shuffle=False, pin_memory=False)
    train_loader_all_warmup = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,drop_last=True, pin_memory=True)
    train_noformdataset_len = train_noformdataset.__len__()
    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    gcd_pretrained = False
    if True:
        projector = DINOHead(in_dim=args.feat_dim, out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers)
        model = Model(args)#nn.Sequential(backbone, projector).to(device)
    if args.dataset_name=='cifar10':
        classnum=10
    elif args.dataset_name=='cifar100' or args.dataset_name=='imagenet_100':
        classnum=100
    elif args.dataset_name=='scars':
        classnum = 196
    elif args.dataset_name=='cub':
        classnum = 200
    else:
        classnum=100
    # ----------------------
    # TRAIN
    # ----------------------
    train(model, train_loader, train_loader_all, test_loader_labelled, test_loader_unlabelled, args, classnum,train_noformdataset_len, train_loader_all_warmup)
