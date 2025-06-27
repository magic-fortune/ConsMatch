import argparse
# from line_profiler import LineProfiler
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from util.thresh_helper import ThreshController, DropRateController
from dataset.acdc import ACDCDataset
from model.unet import UNet
from util.soft_dice_loss import dice_loss
from util.classes import CLASSES
from util.utils import (
    AverageMeter,
    count_params,
    init_log,
    DiceLoss,
    seed_everything,
    DistillationLoss,
    MSELoss
)
from einops import rearrange
from util.dist_helper import setup_distributed
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

def similarity_loss(features):

    features = rearrange(features, 'n c h w -> n c (h w)')
    corr_map = torch.matmul(features, features.transpose(1, 2)) / torch.sqrt(torch.tensor(features.shape[1]).float())
    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)

    batch_size = features.size(0)
    target = torch.eye(batch_size, device=features.device)

    loss = F.mse_loss(similarity_matrix, target)
    
    return loss

def cg_matrix(features):
    features = rearrange(features, 'n c h w -> n c (h w)')
    corr_map = torch.matmul(features, features.transpose(1, 2)) / torch.sqrt(torch.tensor(features.shape[1]).float())
    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)   
    return  similarity_matrix


class Corr2D(nn.Module):
    def __init__(self, nclass=2):
        super(Corr2D, self).__init__()
        self.nclass = nclass

    def forward(self, f1, f2):
        f1 = rearrange(f1, 'n c h w -> n c (h w)')
        f2 = rearrange(f2, 'n c h w -> n c (h w)')
        # print(f1.size(), f2.size())
        corr_map = torch.matmul(f1, f2.transpose(1, 2)) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        return corr_map
    
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size # 创建缓冲区，不会被模型进行优化，但是依旧会保存
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			# 超参数 温度
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
    def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        emb_i, emb_j = emb_i.view(self.batch_size, -1), emb_j.view(self.batch_size, -1)		# (bs, c, h, w)  --->  (bs, c*h*w)
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
    
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


parser = argparse.ArgumentParser(
    description="Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation"
)
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--labeled-id-path", type=str, required=True)
parser.add_argument("--unlabeled-id-path", type=str, required=True)
parser.add_argument("--save-path", type=str, required=True)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--port", default=None, type=int)
parser.add_argument("--eta", default=0.3, type=float)
parser.add_argument('--s1_to_s2', action='store_true', help='s1 supervise s2')
parser.add_argument('--beishu', type=float, default=1.0)
# corr_match_type
parser.add_argument('--corr_match_type', type=str, default='mse', help='correlation match type')
# temperature
parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
parser.add_argument('--only-corr-mt', type=bool, default=False)
parser.add_argument('--only-surp-add', type=bool, default=False)



def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log("global", logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    conf_thresh = cfg["conf_thresh"]
    thresh_controller = ThreshController(
        nclass=4, momentum=0.999 / 4.0, thresh_init=cfg["conf_thresh"]
    )

    drop_rate = cfg["drop_rate"]
    drop_rate_controller = DropRateController(momentum=0.9)

    if rank == 0:
        all_args = {**cfg, **vars(args), "ngpus": world_size}
        logger.info("{}\n".format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)
    
    model = UNet(in_chns=1, class_num=cfg["nclass"])
    if rank == 0:
        logger.info("Total params: {:.1f}M\n".format(count_params(model)))

    optimizer = SGD(model.parameters(), cfg["lr"], momentum=0.9, weight_decay=0.0001, nesterov=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg["nclass"])
    creiterion_kd = DistillationLoss(temp=cfg["T"])
    creiterion_mmd = MSELoss(reduction="none")
    creiterion_contrastive = ContrastiveLoss(batch_size=cfg["batch_size"], device='cuda', temperature=args.temperature)

    trainset_u = ACDCDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_u",
        cfg["crop_size"],
        args.unlabeled_id_path,
    )
    trainset_l = ACDCDataset(
        cfg["dataset"],
        cfg["data_root"],
        "train_l",
        cfg["crop_size"],
        args.labeled_id_path,
        nsample=len(trainset_u.ids),
    )
    valset = ACDCDataset(cfg["dataset"], cfg["data_root"], "val")

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"]*3,
        drop_last=True,
        sampler=trainsampler_l,
    )
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"]*3,
        drop_last=True,
        sampler=trainsampler_u,
    )
    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=cfg["batch_size"],
        pin_memory=True,
        num_workers=cfg["batch_size"]*3,
        drop_last=True,
        sampler=trainsampler_u_mix,
    )
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=3,
        drop_last=False,
        sampler=valsampler,
        pin_memory_device=f"cuda:{local_rank}",
    )

    total_iters = len(trainloader_u) * cfg["epochs"]
    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(args.save_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if rank == 0:
            logger.info("************ Load from checkpoint at epoch %i\n" % epoch)
            
    loss_u_fn = lambda pred, mask, ignore: criterion_dice(
        pred.softmax(dim=1),
        mask.unsqueeze(1).float(),
        ignore=ignore,
    )
    
    loss_x_fn = lambda pred, mask: (criterion_ce(pred, mask) + criterion_dice(pred.softmax(dim=1), mask.unsqueeze(1).float())) / 2.0

    corr = Corr2D(nclass=cfg["nclass"]).cuda()
    
    for epoch in range(epoch + 1, cfg["epochs"]):
        if rank == 0:
            logger.info(
                "===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}, TH: {:.4f}".format(
                    epoch, optimizer.param_groups[0]["lr"], previous_best, conf_thresh
                )
            )

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_corr = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_u_mix.sampler.set_epoch(epoch + cfg["epochs"])

        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        model.train()
        
        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, _, _)
        ) in enumerate(zip(trainloader_l, trainloader_u)):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            
            # bef_time = time.time()

            pred_x_pred_u_w, pred_x_b_pred_u_w_b = model(
                torch.cat((img_x, img_u_w)),ret_feats=True, drop=False
            )
            pred_x, pred_u_w = pred_x_pred_u_w.chunk(2)
            # bottleneck features
            pred_x_bf, pred_u_w_bf = pred_x_b_pred_u_w_b.chunk(2)

            # pred_u_s1, pred_u_s2 = net(torch.cat((img_u_s1, img_u_s2)), ret_feats=True)
            pred_u_s1_pred_u_s2, pred_u_s1_b_pred_u_s2_b = model(
                torch.cat((img_u_s1, img_u_s2)), ret_feats=True, drop=True
            )
            pred_u_s1, pred_u_s2 = pred_u_s1_pred_u_s2.chunk(2)
            # bottleneck features
            pred_u_s1_bf, pred_u_s2_bf = pred_u_s1_b_pred_u_s2_b.chunk(2)
            
            
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            loss_x = loss_x_fn(pred_x, mask_x)

            loss_u_s1 = loss_u_fn(
                pred_u_s1,
                mask_u_w,
                (conf_u_w < conf_thresh).float()
            )
            loss_u_s2 = loss_u_fn(
                pred_u_s2,
                mask_u_w,
                (conf_u_w < conf_thresh).float()
            ) 
            
            if args.s1_to_s2:
                conf_s1 = pred_u_s1.softmax(dim=1).max(dim=1)[0]
                mask_s1 = pred_u_s1.argmax(dim=1).detach()
                loss_s1_s2 = loss_u_fn(
                    pred_u_s2,
                    mask_s1,
                    (conf_s1 < conf_thresh).float()
                )
                
            # corr_u_w_s1 = corr(pred_u_w_bf, pred_u_s1_bf)
            # corr_u_w_s2 = corr(pred_u_w_bf, pred_u_s2_bf)
            
            # if args.corr_match_type == 'mse':
            #     loss_corr_mse = F.mse_loss(corr_u_w_s1, corr_u_w_s2)
            # elif args.corr_match_type == 'kl':
            #     # with T
            #     loss_corr_mse = (
            #         F.kl_div(F.log_softmax(corr_u_w_s1 / args.temperature, dim=1), F.softmax(corr_u_w_s2 / args.temperature, dim=1)) +
            #         F.kl_div(F.log_softmax(corr_u_w_s2 / args.temperature, dim=1), F.softmax(corr_u_w_s1 / args.temperature, dim=1))
            #     ) / 2.0
                
            
            # label_corr = torch.arange(corr_u_w_s1.size(1)).unsqueeze(0).repeat(corr_u_w_s1.size(0), 1).cuda()
            # loss_corr_u_w_s1 = F.cross_entropy(corr_u_w_s1, label_corr)
            # loss_corr_u_w_s2 = F.cross_entropy(corr_u_w_s2, label_corr)
            
            # loss_corr_u_w_s1 = creiterion_contrastive(pred_u_w_bf, pred_u_s1_bf)
            # loss_corr_u_w_s2 = creiterion_contrastive(pred_u_w_bf, pred_u_s2_bf)
            
            # loss_corr_cont = (loss_corr_u_w_s1 + loss_corr_u_w_s2) / 2.0
            
            # loss_corr = args.eta * loss_corr_mse + (1 - args.eta) * loss_corr_cont
            pred_u_w_bf_mt = cg_matrix(pred_u_w_bf)
            pred_u_s1_bf_mt = cg_matrix(pred_u_s1_bf)
            pred_u_s2_bf_mt = cg_matrix(pred_u_s2_bf)
                        
            loss_corr_mt = (similarity_loss(pred_u_w_bf) + similarity_loss(pred_u_s1_bf) + similarity_loss(pred_u_s2_bf)) * 0.333
            if args.corr_match_type == 'mse':
                loss_surp_add =  (F.mse_loss(pred_u_s1_bf_mt, pred_u_w_bf_mt) + F.mse_loss(pred_u_s2_bf_mt, pred_u_w_bf_mt) + F.mse_loss(pred_u_s2_bf_mt, pred_u_s1_bf_mt)) * 0.33
            else:
                loss_surp_add = (
                    F.kl_div(F.log_softmax(pred_u_s1_bf_mt / args.temperature, dim=1), F.softmax(pred_u_w_bf_mt / args.temperature, dim=1)) +
                    F.kl_div(F.log_softmax(pred_u_w_bf_mt / args.temperature, dim=1), F.softmax(pred_u_s1_bf_mt / args.temperature, dim=1)) + 
                    F.kl_div(F.log_softmax(pred_u_s2_bf_mt / args.temperature, dim=1), F.softmax(pred_u_w_bf_mt / args.temperature, dim=1)) +
                    F.kl_div(F.log_softmax(pred_u_w_bf_mt / args.temperature, dim=1), F.softmax(pred_u_s2_bf_mt / args.temperature, dim=1)) + 
                    F.kl_div(F.log_softmax(pred_u_s2_bf_mt / args.temperature, dim=1), F.softmax(pred_u_s1_bf_mt / args.temperature, dim=1)) +
                    F.kl_div(F.log_softmax(pred_u_s1_bf_mt / args.temperature, dim=1), F.softmax(pred_u_s2_bf_mt / args.temperature, dim=1)) ) / 6.0
            
            
            if args.only_corr_mt:
                loss_corr =  loss_surp_add 
            else:
                loss_corr = (loss_corr_mt + loss_surp_add) * 0.5
            if args.only_surp_add:
                loss_corr =  loss_surp_add 
            else:
                loss_corr = (loss_corr_mt + loss_surp_add) * 0.5
            
            
            
            if args.s1_to_s2:
                loss = (loss_x   + ((loss_u_s1 + loss_u_s2 + loss_s1_s2))  * (1 - args.beishu) + loss_corr * args.beishu) / 2.0
            else:
                loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_corr * 0.5) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_corr.update(loss_corr.item())

            mask_ratio = (conf_u_w >= conf_thresh).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i

            lr = cfg["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if cfg["use_threshold_relax"]:
                thresh_controller.thresh_update(
                    pred_u_w.detach().float(),
                    None,
                    update_g=True,
                )
                conf_thresh = thresh_controller.get_thresh_global()

            if rank == 0:
                writer.add_scalar("train/loss_all", loss.item(), iters)
                writer.add_scalar("train/loss_x", loss_x.item(), iters)
                writer.add_scalar(
                    "train/loss_s", (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters
                )
                writer.add_scalar("train/loss_corr", loss_corr.item(), iters)
                writer.add_scalar("train/mask_ratio", mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    "Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss KD: {:.3f}, Mask ratio: {:.3f}".format(
                        i,
                        total_loss.avg,
                        total_loss_x.avg,
                        total_loss_s.avg,
                        total_loss_corr.avg,
                        total_mask_ratio.avg
                    )
                )

        model.eval()

        dice_class = [0] * 3

        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(
                    img,
                    (cfg["crop_size"], cfg["crop_size"]),
                    mode="bilinear",
                    align_corners=False,
                )

                img = img.permute(1, 0, 2, 3)

                pred = model(img)

                pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg["nclass"]):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls - 1] += 2.0 * inter / union

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        mean_dice = sum(dice_class) / len(dice_class)

        if rank == 0:
            for cls_idx, dice in enumerate(dice_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] Dice: "
                    "{:.2f}".format(cls_idx, CLASSES[cfg["dataset"]][cls_idx], dice)
                )
            logger.info(
                "***** Evaluation ***** >>>> MeanDice: {:.2f}\n".format(mean_dice)
            )

            writer.add_scalar("eval/MeanDice", mean_dice, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar(
                    "eval/%s_dice" % (CLASSES[cfg["dataset"]][i]), dice, epoch
                )

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))


if __name__ == "__main__":
    main()
