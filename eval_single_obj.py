# ------------------------------------------------------------------------
# Yuanwen Yue
# ETH Zurich
# ------------------------------------------------------------------------

import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model, build_criterion
import MinkowskiEngine as ME
from utils.seg import mean_iou_scene, extend_clicks, get_simulated_clicks
import utils.misc as utils
from datetime import datetime

from evaluation.evaluator_SO import EvaluatorSO
import wandb
import os

from plukovic.simulate_clicks import get_simualted_clicks_scene_sam
from plukovic.scannet_scene import SensorData
from plukovic.visualisation import visualize_iou_scene, visualize_gt_scene

def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)

    # dataset
    parser.add_argument('--dataset', default='scannet')
    parser.add_argument('--dataset_mode', default='single_obj')
    parser.add_argument('--scan_folder', default='data/ScanNet/scans', type=str)
    parser.add_argument('--crop', default=False, action='store_true', help='whether evaluate on whole scan or object crops')
    parser.add_argument('--val_list', default='data/ScanNet/single/object_ids.npy', type=str)
    parser.add_argument('--val_list_classes', default='data/ScanNet/single/object_classes.txt', type=str)
    parser.add_argument('--train_list', default='', type=str)
    
    # model
    ### 1. backbone
    parser.add_argument('--dialations', default=[ 1, 1, 1, 1 ], type=list)
    parser.add_argument('--conv1_kernel_size', default=5, type=int)
    parser.add_argument('--bn_momentum', default=0.02, type=int)
    parser.add_argument('--voxel_size', default=0.05, type=float)

    ### 2. transformer
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_decoders', default=3, type=int)
    parser.add_argument('--num_bg_queries', default=10, type=int, help='number of learnable background queries')
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--pre_norm', default=False, type=bool)
    parser.add_argument('--normalize_pos_enc', default=True, type=bool)
    parser.add_argument('--positional_encoding_type', default="fourier", type=str)
    parser.add_argument('--gauss_scale', default=1.0, type=float, help='gauss scale for positional encoding')
    parser.add_argument('--hlevels', default=[4], type=list)
    parser.add_argument('--shared_decoder', default=False, type=bool)
    parser.add_argument('--aux', default=True, type=bool)

    # evaluation
    parser.add_argument('--val_batch_size', default=1, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', default='checkpoints/checkpoint1099.pth', help='resume from checkpoint')

    parser.add_argument('--max_num_clicks', default=20, help='maximum number of clicks per object on average', type=int)

    return parser



def Evaluate(model, data_loader, args, device):
    model.eval()
    args = parser.parse_args()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    instance_counter = 0
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'val_results_single_{timestamp}.csv'
    results_file = os.path.join(args.output_dir, filename)
    f = open(results_file, 'w')

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        coords, raw_coords, feats, labels, labels_full, inverse_map, click_idx, scene_name, object_id = batched_inputs
        sensors = SensorData(os.path.join(args.scan_folder, scene_name[0], scene_name[0] + '.hdf5'))

        coords = coords.to(device)
        raw_coords = raw_coords.to(device)
        labels = [l.to(device) for l in labels]
        labels_full = [l.to(device) for l in labels_full]

        data = ME.SparseTensor(
                                coordinates=coords,
                                features=feats,
                                device=device
                                )

        ###### interactive evaluation ######
        batch_idx = coords[:,0]
        batch_size = batch_idx.max()+1

        # click ids set null
        click_idx = [{'0':[],'1':[]} for b in range(batch_size)]

        click_time_idx = copy.deepcopy(click_idx)

        # pre-compute backbone features only once
        pcd_features, aux, coordinates, pos_encodings_pcd = model.forward_backbone(data, raw_coordinates=raw_coords)

        sim_clicks = []
        sim_click_times = []

        for idx in range(batch_size):

            pred = [torch.zeros(l.shape).to(device) for l in labels]
            sample_mask = batch_idx == idx
            sample_pred = pred[idx]
            sample_labels = labels[idx]
            sample_raw_coords = raw_coords[sample_mask]

            new_clicks, _, _, new_click_times = get_simulated_clicks(sample_pred, sample_labels, sample_raw_coords, 0, training=False)
            print(f"-----------------------------------------------------------------------------------------")
            sam_click_list, sam_click_times_list = get_simualted_clicks_scene_sam(raw_coords, new_clicks, sensors)
            print(f"-----------------------------------------------------------------------------------------")
            
            sim_clicks.append([])
            sim_clicks[idx].extend([new_clicks])
            sim_clicks[idx].extend(sam_click_list)

            sim_click_times.append([])
            sim_click_times[idx].extend([new_click_times])
            sim_click_times[idx].extend(sam_click_times_list)

        for current_num_clicks in range(len(sim_clicks[0])+1):

            if current_num_clicks == 0:
                pred = [torch.zeros(l.shape).to(device) for l in labels]
            else:
                outputs = model.forward_mask(pcd_features, aux, coordinates, pos_encodings_pcd,
                                             click_idx=click_idx, click_time_idx=click_time_idx)
                pred_logits = outputs['pred_masks']
                pred = [p.argmax(-1) for p in pred_logits]

            updated_pred = []

            for idx in range(batch_size):
        
                sample_mask = batch_idx == idx
                sample_pred = pred[idx]

                if current_num_clicks != 0:
                    # update prediction with sparse gt
                    for obj_id, cids in click_idx[idx].items():
                        sample_pred[cids] = int(obj_id)
                    updated_pred.append(sample_pred)

                sample_labels = labels[idx]
                sample_raw_coords = raw_coords[sample_mask]
                sample_pred_full = sample_pred[inverse_map[idx]]

                if current_num_clicks >= 1:
                    visualize_iou_scene(raw_coords, sample_pred, sample_labels)
                    visualize_gt_scene(raw_coords, sample_labels)

                sample_labels_full = labels_full[idx]
                sample_iou, _ = mean_iou_scene(sample_pred_full, sample_labels_full)

                line = str(instance_counter+idx) + ' ' + scene_name[idx].replace('scene','') + ' '  + object_id[idx] + ' ' + str(current_num_clicks) +  ' ' + str(
                sample_iou.cpu().numpy()) + '\n'
                f.write(line)
                print(scene_name[idx], ' | Object: ', object_id[idx], ' | num clicks: ', current_num_clicks, ' | IOU: ', sample_iou.item())

                if current_num_clicks < len(sim_clicks[idx]):
                    new_clicks = sim_clicks[idx][current_num_clicks]
                    new_click_time = sim_click_times[idx][current_num_clicks]
                    print(f"Adding new clicks: {new_clicks}.")
                    click_idx[idx], click_time_idx[idx] = extend_clicks(click_idx[idx], click_time_idx[idx], new_clicks, new_click_time)
                    
                else:
                    print(f"No more clicks to add.")

                print(f"-----------------------------------------------------------------------------------------")

    f.close()
    #evaluator = EvaluatorSO(args.dataset, args.val_list, args.val_list_classes, results_file, [0.5,0.65,0.8,0.85,0.9])
    #results_dict = evaluator.eval_results()

def main(args):

    #device = torch.device(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model = build_model(args)
    model.to(device)

    # build dataset and dataloader
    dataset_val, collation_fn_val = build_dataset(split='val', args=args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, args.val_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collation_fn_val, num_workers=args.num_workers,
                                 pin_memory=True)

    output_dir = Path(args.output_dir)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))
      
    Evaluate(model, data_loader_val, args, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation script on interactive multi-object segmentation ', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
