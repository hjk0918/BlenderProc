import subprocess
import sys
import json
import os
from tqdm import tqdm

scene_dir = '/data/bhuai/BlenderProc/FRONT3D_render'
scenes = sorted(os.listdir(scene_dir))

layout_dir = '/data/bhuai/3D-FRONT'
scene_list = sorted(os.listdir(layout_dir))

output_dir = '/data/bhuai/BlenderProc/FRONT3D_render'
existed_scenes = sorted(os.listdir(output_dir))

for scene in tqdm(scenes):
    scene_idx = int(scene.split('_')[1])
    room_idx = int(scene.split('_')[2])

    print(f"Rendering depth and segmentation map for scene {scene}")

    bashCommand = f"python cli.py run ./scripts/render.py -s {scene_idx} -r {room_idx} " \
                  f"--mode seg --gpu 7 --pose_dir {scene_dir}"
    process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=sys.stdout)
    output, error = process.communicate()
