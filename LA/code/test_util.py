import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict

import math
import numpy as np
import torch
from torch.nn import functional as F
import os
import hashlib
import time
from multiprocessing import Pool


imgs_cache_list = []
lab_cache_list = []

single_case_cache = {}
single_case_idx = 0

fast_mode = False

# def test_all_case(
#     net,
#     image_list,
#     num_classes,
#     patch_size=(112, 112, 80),
#     stride_xy=18,
#     stride_z=4,
#     save_result=True,
#     test_save_path=None,
#     preproc_fn=None,
# ):
#     total_metric = 0.0
#     metric_dict = OrderedDict()
#     metric_dict["name"] = list()
#     metric_dict["dice"] = list()
#     metric_dict["jaccard"] = list()
#     metric_dict["asd"] = list()
#     metric_dict["95hd"] = list()
#     for idx, image_path in enumerate(tqdm(image_list)):
#         case_name = image_path.split("/")[-2]
#         id = image_path.split("/")[-1]
        
#         if len(imgs_cache_list) > idx:
#             image = imgs_cache_list[idx]
#             label = lab_cache_list[idx]
#         else:
#             h5f = h5py.File(image_path, "r")
#             image = h5f["image"][:]
#             label = h5f["label"][:]
#             if preproc_fn is not None:
#                 image = preproc_fn(image)
#             imgs_cache_list.append(image)
#             lab_cache_list.append(label)
        
#         prediction, score_map = test_single_case(
#             net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, idx=str(idx)
#         )
#         a = time.time()
#         if np.sum(prediction) == 0:
#             single_metric = (0, 0, 0, 0)
#         else:
#             single_metric = calculate_metric_percase(prediction, label[:])
#             metric_dict["name"].append(case_name)
#             metric_dict["dice"].append(single_metric[0])
#             metric_dict["jaccard"].append(single_metric[1])
#             metric_dict["asd"].append(single_metric[2])
#             metric_dict["95hd"].append(single_metric[3])
#             # print(metric_dict)
#         print(time.time() - a)
#         total_metric += np.asarray(single_metric)

#         if save_result:
#             test_save_path_temp = os.path.join(test_save_path, case_name)
#             if not os.path.exists(test_save_path_temp):
#                 os.makedirs(test_save_path_temp)
#             nib.save(
#                 nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
#                 test_save_path_temp + "/" + id + "_pred.nii.gz",
#             )
#             nib.save(
#                 nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
#                 test_save_path_temp + "/" + id + "_img.nii.gz",
#             )
#             nib.save(
#                 nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
#                 test_save_path_temp + "/" + id + "_gt.nii.gz",
#             )
#     avg_metric = total_metric / len(image_list)
#     if save_result:
#         metric_csv = pd.DataFrame(metric_dict)
#         metric_csv.to_csv(test_save_path + "/metric.csv", index=False)
#     print("average metric is {}".format(avg_metric))

#     return avg_metric


def compute_metric(args):
    prediction, label = args
    if torch.sum(prediction) == 0:
        return (0, 0, 0, 0)
    else:
        return calculate_metric_percase(prediction, label, fast_mode=fast_mode)

def test_all_case(
    net,
    image_list,
    num_classes,
    patch_size=(112, 112, 80),
    stride_xy=18,
    stride_z=4,
    save_result=True,
    test_save_path=None,
    preproc_fn=None,
):
    total_metric = 0.0
    metric_dict = OrderedDict()
    metric_dict["name"] = list()
    metric_dict["dice"] = list()
    metric_dict["jaccard"] = list()
    metric_dict["asd"] = list()
    metric_dict["95hd"] = list()

    predictions_list = []
    labels_list = []
    case_names_list = []

    for idx, image_path in enumerate(tqdm(image_list)):
        case_name = image_path.split("/")[-2]
        id = image_path.split("/")[-1]

        if len(imgs_cache_list) > idx:
            image = imgs_cache_list[idx]
            label = lab_cache_list[idx]
        else:
            with h5py.File(image_path, "r") as h5f:
                image = h5f["image"][:]
                label = h5f["label"][:]
            if preproc_fn is not None:
                image = preproc_fn(image)
            imgs_cache_list.append(image)
            lab_cache_list.append(label)

        prediction, score_map = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, idx=str(idx)
        )

        # Collect predictions, labels, and case names for parallel processing
        predictions_list.append(prediction)
        labels_list.append(torch.from_numpy(label).cuda().long())
        case_names_list.append(case_name)

        # if save_result:
        #     test_save_path_temp = os.path.join(test_save_path, case_name)
        #     os.makedirs(test_save_path_temp, exist_ok=True)
        #     nib.save(
        #         nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
        #         os.path.join(test_save_path_temp, f"{id}_pred.nii.gz"),
        #     )
        #     nib.save(
        #         nib.Nifti1Image(image.astype(np.float32), np.eye(4)),
        #         os.path.join(test_save_path_temp, f"{id}_img.nii.gz"),
        #     )
        #     nib.save(
        #         nib.Nifti1Image(label.astype(np.float32), np.eye(4)),
        #         os.path.join(test_save_path_temp, f"{id}_gt.nii.gz"),
        #     )

    # Compute metrics in parallel
    # with Pool(processes=24) as pool:
    #     metrics = pool.map(compute_metric, zip(predictions_list, labels_list))
    
    metrics = [compute_metric(args) for args in zip(predictions_list, labels_list)]
    # for prediction, label in zip(predictions_list, labels_list):
        
    # Update metric dictionary and total metrics
    for case_name, single_metric in zip(case_names_list, metrics):
        metric_dict["name"].append(case_name)
        metric_dict["dice"].append(single_metric[0])
        metric_dict["jaccard"].append(single_metric[1])
        metric_dict["asd"].append(single_metric[2])
        metric_dict["95hd"].append(single_metric[3])
        total_metric += np.asarray(single_metric)

    avg_metric = total_metric / len(image_list)
    if save_result:
        metric_csv = pd.DataFrame(metric_dict)
        metric_csv.to_csv(os.path.join(test_save_path, "metric.csv"), index=False)
    print("average metric is {}".format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, idx='0'):
    if idx not in single_case_cache:
        single_case_cache[idx] = []
    
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        print('padding')
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    # score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    score_map = torch.zeros((num_classes,) + image.shape).cuda()
    # cnt = np.zeros(image.shape).astype(np.float32)
    cnt = torch.zeros(image.shape).cuda()
    
    idx_int = -1
    create_cache = True
    if len(single_case_cache[idx]) > 0:
        create_cache = False
    
    if not create_cache:
        with torch.no_grad():
            y1 = []
            for i in range(len(single_case_cache[idx])):
                y1.append(net(single_case_cache[idx][i]))
            y1 = torch.cat(y1, dim=0)
            # y1 = net(test_patch.cuda()).cpu()
            y = F.softmax(y1, dim=1)
        # all_y = [y[i].cpu().data.numpy() for i in range(test_patch.size(0))]
            all_y = y.chunk(y1.size(0), dim=0)
    
    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                idx_int += 1
                if not create_cache:
                    test_patch = single_case_cache[idx][idx_int]
                    y = all_y[idx_int]
                    y = y[0, :, :, :, :]
                else:
                    test_patch = image[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    test_patch = np.expand_dims(
                        np.expand_dims(test_patch, axis=0), axis=0
                    ).astype(np.float32)
                    test_patch = torch.from_numpy(test_patch).cuda()
                    single_case_cache[idx].append(test_patch)
                    
                    with torch.no_grad():
                        y1 = net(test_patch)
                        y = F.softmax(y1, dim=1)
                        # y = y.cpu().data.numpy()
                        y = y[0, :, :, :, :]
                score_map[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    score_map[
                        :,
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + y
                )
                cnt[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    cnt[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + 1
                )
    
    # score_map = score_map / np.expand_dims(cnt, axis=0)
    score_map = score_map / cnt.unsqueeze(0)
    # label_map = np.argmax(score_map, axis=0)
    label_map = torch.argmax(score_map, dim=0)
    if add_pad:
        label_map = label_map[
            wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
        score_map = score_map[
            :, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
    return label_map, score_map

# def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    

#     # Define cache directory
#     cache_dir = './cache_ram'
#     if not os.path.exists(cache_dir):
#         os.makedirs(cache_dir)

#     # Generate a unique hash for the image based on its content
#     image_hash = hashlib.md5(image.tobytes()).hexdigest()
#     # Create a cache key using image hash and patching parameters
#     cache_key = f"{image_hash}_{patch_size[0]}_{patch_size[1]}_{patch_size[2]}_{stride_xy}_{stride_z}"
#     # Cache file path
#     cache_file = os.path.join(cache_dir, f"{cache_key}.npz")

#     # Check if the cache exists
#     if os.path.exists(cache_file):
#         print(f"Loading patches from cache {cache_file}")
#         data = np.load(cache_file)
#         test_patches = data['test_patches']
#         xs_list = data['xs_list']
#         ys_list = data['ys_list']
#         zs_list = data['zs_list']
#         add_pad = data['add_pad'].item()
#         w_pad = data['w_pad'].item()
#         h_pad = data['h_pad'].item()
#         d_pad = data['d_pad'].item()
#         wl_pad = data['wl_pad'].item()
#         hl_pad = data['hl_pad'].item()
#         dl_pad = data['dl_pad'].item()
#         wr_pad = data['wr_pad'].item()
#         hr_pad = data['hr_pad'].item()
#         dr_pad = data['dr_pad'].item()
#         w = data['w'].item()
#         h = data['h'].item()
#         d = data['d'].item()
#         ww = data['ww'].item()
#         hh = data['hh'].item()
#         dd = data['dd'].item()
#     else:
#         print("Cache not found; processing and caching patches.")
#         w, h, d = image.shape

#         # Padding logic (unchanged)
#         add_pad = False
#         if w < patch_size[0]:
#             w_pad = patch_size[0] - w
#             add_pad = True
#         else:
#             w_pad = 0
#         if h < patch_size[1]:
#             h_pad = patch_size[1] - h
#             add_pad = True
#         else:
#             h_pad = 0
#         if d < patch_size[2]:
#             d_pad = patch_size[2] - d
#             add_pad = True
#         else:
#             d_pad = 0
#         wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
#         hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
#         dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
#         if add_pad:
#             image = np.pad(
#                 image,
#                 [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
#                 mode="constant",
#                 constant_values=0,
#             )
#         ww, hh, dd = image.shape

#         sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
#         sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
#         sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

#         # Initialize empty lists to store patches and positions
#         test_patches = []
#         xs_list = []
#         ys_list = []
#         zs_list = []

#         # Extract patches and store them along with their positions
#         for x in range(0, sx):
#             xs = min(stride_xy * x, ww - patch_size[0])
#             for y in range(0, sy):
#                 ys = min(stride_xy * y, hh - patch_size[1])
#                 for z in range(0, sz):
#                     zs = min(stride_z * z, dd - patch_size[2])
#                     test_patch = image[
#                         xs : xs + patch_size[0],
#                         ys : ys + patch_size[1],
#                         zs : zs + patch_size[2],
#                     ]
#                     test_patches.append(test_patch)
#                     xs_list.append(xs)
#                     ys_list.append(ys)
#                     zs_list.append(zs)

#         # Save patches and necessary variables to cache
#         np.savez_compressed(
#             cache_file,
#             test_patches=np.array(test_patches),
#             xs_list=np.array(xs_list),
#             ys_list=np.array(ys_list),
#             zs_list=np.array(zs_list),
#             add_pad=add_pad,
#             w_pad=w_pad,
#             h_pad=h_pad,
#             d_pad=d_pad,
#             wl_pad=wl_pad,
#             hl_pad=hl_pad,
#             dl_pad=dl_pad,
#             wr_pad=wr_pad,
#             hr_pad=hr_pad,
#             dr_pad=dr_pad,
#             w=w,
#             h=h,
#             d=d,
#             ww=ww,
#             hh=hh,
#             dd=dd,
#         )

#     # Initialize score_map and cnt
#     score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
#     cnt = np.zeros(image.shape).astype(np.float32)

#     num_patches = len(test_patches)
#     for i in range(num_patches):
#         test_patch = test_patches[i]
#         xs = xs_list[i]
#         ys = ys_list[i]
#         zs = zs_list[i]

#         # Process the test_patch
#         test_patch = np.expand_dims(
#             np.expand_dims(test_patch, axis=0), axis=0
#         ).astype(np.float32)
#         test_patch = torch.from_numpy(test_patch).cuda()
#         y1 = net(test_patch)
#         y = F.softmax(y1, dim=1)
#         y = y.cpu().data.numpy()
#         y = y[0, :, :, :, :]

#         # Accumulate the probability maps
#         score_map[
#             :,
#             xs : xs + patch_size[0],
#             ys : ys + patch_size[1],
#             zs : zs + patch_size[2],
#         ] += y

#         # Increment the counter for averaging
#         cnt[
#             xs : xs + patch_size[0],
#             ys : ys + patch_size[1],
#             zs : zs + patch_size[2],
#         ] += 1

#     # Average the score_map to get final predictions
#     score_map = score_map / np.expand_dims(cnt, axis=0)
#     label_map = np.argmax(score_map, axis=0)

#     # Remove padding if added
#     if add_pad:
#         label_map = label_map[
#             wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
#         ]
#         score_map = score_map[
#             :, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
#         ]

#     return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = prediction == i
        label_tmp = label == i
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = (
            2
            * np.sum(prediction_tmp * label_tmp)
            / (np.sum(prediction_tmp) + np.sum(label_tmp))
        )
        total_dice[i - 1] += dice

    return total_dice


from torchmetrics.segmentation import HausdorffDistance

def cal_dc(result, reference):
    result = result.bool().reshape(-1)
    reference = reference.bool().reshape(-1)
    
    intersection = torch.sum(result & reference).float()
    
    size_i1 = torch.sum(result).float()
    size_i2 = torch.sum(reference).float()
    
    dc_denominator = size_i1 + size_i2
    if dc_denominator.item() == 0:
        dc = 0.0
    else:
        dc = 2. * intersection / dc_denominator
    
    return dc

def cal_jc(result, reference):
    result = result.bool().reshape(-1)
    reference = reference.bool().reshape(-1)
    
    intersection = torch.sum(result & reference).float()
    union = torch.sum(result | reference).float()
    
    if union.item() == 0:
        jc = 0.0
    else:
        jc = intersection / union
    
    return jc


def calculate_metric_percase(pred, gt, fast_mode=False):
    if fast_mode:
        dice = cal_dc(pred, gt).item()
        jc = cal_jc(pred, gt).item()
        hd, asd = jc / 10, jc / 10
    else:
        pred = pred.cpu().numpy()
        gt = gt.cpu().numpy()
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)

    return dice, jc, hd, asd
