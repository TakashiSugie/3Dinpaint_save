import numpy as np
import argparse
import os
import vispy
from tqdm import tqdm
import yaml
import time
from meshPly import write_ply, read_ply, output_3d_photo
from utils import get_MiDaS_samples, read_MiDaS_depth
import cv2
import sys
import copy
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',
                    help='Configure of post processing')
parser.add_argument('--filePath1', type=str, default='',
                    help='processing file name')
parser.add_argument('--filePath2', type=str, default='',
                    help='processing file name')


args = sys.argv
if len(args) < 4:
    print("please")
    pathList = glob.glob("./image/*png")
    img1Path = pathList[1]
    img2Path = pathList[0]


args = parser.parse_args()
if args.filePath2:
    img1Path = args.filePath1
    img2Path = args.filePath2


imgName1 = os.path.splitext(os.path.basename(img1Path))[0]
imgName2 = os.path.splitext(os.path.basename(img2Path))[0]

config = yaml.load(open(args.config, 'r'))
if config['offscreen_rendering'] is True:
    vispy.use(app='egl')
os.makedirs(config['mesh_folder'], exist_ok=True)
os.makedirs(config['video_folder'], exist_ok=True)
os.makedirs(config['depth_folder'], exist_ok=True)
sample_list = get_MiDaS_samples(
    img1Path, config['src_folder'], config['depth_folder'], config, config['specific'])
normal_canvas, all_canvas = None, None

if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
    device = config["gpu_ids"]
else:
    device = "cpu"

print(f"running on device {device}")
plyPath = "new_"+imgName1+"_"+imgName2

for idx in tqdm(range(len(sample_list))):
    sample = sample_list[idx]
    mesh_fi = os.path.join(config['mesh_folder'],
                           # sample['src_pair_name'] + '.ply')
                           plyPath+".ply")
    config['output_h'], config['output_w'] = 640, 640
    mean_loc_depth = 0.7425095  # この値は決め打ちでいってる
    # これは、デプスマップの中心のデプスの値

    if config['save_ply'] is True or config['load_ply'] is True:
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    print(f"Making video at {time.time()}")
    videos_poses, video_basename = copy.deepcopy(
        sample['tgts_poses']), sample['tgt_name']
    top = (config.get('output_h') // 2 -
           sample['int_mtx'][1, 2] * config['output_h'])
    left = (config.get('output_w') // 2 -
            sample['int_mtx'][0, 2] * config['output_w'])
    down, right = top + config['output_h'], left + config['output_w']
    border = [int(xx) for xx in [top, down, left, right]]
    normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov),
                                                copy.deepcopy(sample['tgt_pose']), sample['video_postfix'], copy.deepcopy(
                                                    sample['ref_pose']), copy.deepcopy(config['video_folder']),
                                                copy.deepcopy(
                                                sample['int_mtx']),
                                                config,
                                                videos_poses, video_basename, config.get('output_h'), config.get('output_w'), border=border, normal_canvas=normal_canvas, all_canvas=all_canvas, mean_loc_depth=mean_loc_depth)
