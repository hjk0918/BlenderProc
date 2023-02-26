import subprocess
import sys
import json
import os

with open('/data/bhuai/front3d_blender/render_plan.json', 'r') as f:
    render_plan = json.load(f)

layout_dir = '/data/bhuai/3D-FRONT'
scene_list = sorted(os.listdir(layout_dir))

output_dir = '/data/bhuai/BlenderProc/FRONT3D_render'
existed_scenes = sorted(os.listdir(output_dir))

scenes = list(render_plan.keys())[::-1]

for scene in scenes:
    rooms = render_plan[scene]['rooms']
    scene_idx = int(scene.split('_')[-1])

    if scene_list[scene_idx].split('.')[0] != render_plan[scene]['scene_id']:
        print(f"Scene id mismatch: {scene_list[scene_idx]} {render_plan[scene]['scene_id']}")
        continue

    for r, room in enumerate(rooms):
        if f'3dfront_{scene_idx:04d}_{r:02d}' in existed_scenes:
            print(f"3dfront_{scene_idx:04d}_{r:02d} already exists.")
            continue

        print(f"Rendering scene {scene}, room {r}: {room}")

        bashCommand = f"python cli.py run ./scripts/render.py -s {scene_idx} -r {r} --mode render -nc --gpu 7"
        process = subprocess.Popen(bashCommand.split(), stderr=sys.stderr, stdout=sys.stdout)
        output, error = process.communicate()
