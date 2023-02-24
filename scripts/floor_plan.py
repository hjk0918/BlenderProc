import blenderproc as bproc
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
import json

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'scripts'))

from load_helper import *
from render_configs import *

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)


class FloorPlan():
    def __init__(self, json_path, loaded_objects=None):
        self.json_path = json_path
        with open(json_path) as f:
            self.scene_json = json.load(f)

        if loaded_objects is None:
            self.objects = load_scene_objects(json_path)
        else:
            self.objects = loaded_objects
        self.names, self.bboxes, self.uids, self.obj_room_ids = get_scene_rot_bbox_meta(self.objects)

        self.bbox_mins = np.min(self.bboxes, axis=1)
        self.bbox_maxs = np.max(self.bboxes, axis=1)

        self.scene_min = np.min(self.bbox_mins, axis=0)
        self.scene_max = np.max(self.bbox_maxs, axis=0)
        print('scene_min: ', self.scene_min)
        print('scene_max: ', self.scene_max)

        if (self.scene_max - self.scene_min > 100).any():
            print('scene too large, skip')

        self.scale = 200
        self.margin = 100

        self.width = int((self.scene_max - self.scene_min)[0] * self.scale) + self.margin * 2
        self.height = int((self.scene_max - self.scene_min)[1] * self.scale) + self.margin * 2
        self.image = np.ones((self.height, self.width, 3), np.uint8)

        self.colors = np.multiply([
            plt.cm.get_cmap('gist_ncar', 37)((i * 7 + 5) % 37)[:3] for i in range(37)
        ], 255).astype(np.uint8)

        self.ignore_types = ['wall', 'floor', 'ceiling', 'slab', 'baseboard', 'door', 'window', 
            'pocket', 'front', 'back']

        self.room_meta = None

    def draw_coords(self):
        seg = 0.08
        x0, y0 = self.point_to_image([0, 0, 0])
        cv2.line(self.image, (0, y0), (self.width-1, y0),
                 color=red, thickness=3)
        cv2.line(self.image, (x0, 0), (x0, self.height-1),
                 color=red, thickness=3)

        for i in range(int(np.floor(self.scene_min[0])), int(np.ceil(self.scene_max[0])+1)):
            cv2.line(self.image, self.point_to_image(
                [i, -seg]), self.point_to_image([i, seg]), color=red, thickness=2)
        for i in range(int(np.floor(self.scene_min[1])), int(np.ceil(self.scene_max[1])+1)):
            cv2.line(self.image, self.point_to_image(
                [-seg, i]), self.point_to_image([seg, i]), color=red, thickness=2)

        cv2.putText(self.image, 'x+', (self.width-80, y0-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)
        cv2.putText(self.image, 'y+', (x0+20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, red, thickness=2)

    def get_room_meta(self):
        if self.room_meta is None:
            self.draw_room_bbox()
        return self.room_meta

    def draw_room_bbox(self, margin=0.1):
        rooms = self.scene_json['scene']['room']
        self.room_meta = []
        for i, room in enumerate(rooms):
            room_id = room['instanceid']
            children = room['children']
            uids = [c['ref'] for c in children]

            box_min = []        # do not consider floor, ceiling, slab, baseboard, etc.
            box_max = []
            box_max_all = []    # consider all objects
            box_min_all = []

            for uid, obj_room_id, bbox, name in zip(self.uids, self.obj_room_ids, self.bboxes, self.names):
                ignored = False
                for ignore_type in self.ignore_types:
                    if ignore_type in name.lower():
                        ignored = True
                        break
                
                if not np.isfinite(bbox).all():
                    continue

                if uid in uids and (obj_room_id == i or obj_room_id == -1):
                    if not ignored:
                        box_min.append(np.min(bbox, axis=0))
                        box_max.append(np.max(bbox, axis=0))
                        # print('room {} has object {}'.format(room_id, name))

                    box_min_all.append(np.min(bbox, axis=0))
                    box_max_all.append(np.max(bbox, axis=0))

            if len(box_min) == 0:
                print('room {} has no objects'.format(room_id))
                continue

            box_min = np.min(box_min, axis=0)
            box_max = np.max(box_max, axis=0)
            diag = box_max - box_min
            box_min -= diag * margin
            box_max += diag * margin

            box_min_all = np.min(box_min_all, axis=0)
            box_max_all = np.max(box_max_all, axis=0)

            # get the intersection of room bbox and all objects bbox
            inter_min = np.maximum(box_min, box_min_all)
            inter_max = np.minimum(box_max, box_max_all)
            union_min = np.minimum(box_min, box_min_all)
            union_max = np.maximum(box_max, box_max_all)

            x_scale = (union_max[0] - union_min[0]) / (inter_max[0] - inter_min[0])
            y_scale = (union_max[1] - union_min[1]) / (inter_max[1] - inter_min[1])
            if x_scale > 1.5 or y_scale > 1.5:
                # use intersection
                box_min = inter_min
                box_max = inter_max
            else:
                box_min = box_min_all
                box_max = box_max_all

            box_min[2] = box_min_all[2]     # use the floor height
            box_max[2] = box_max_all[2]     # use the ceiling height

            cv2.rectangle(self.image, self.point_to_image(box_min),
                          self.point_to_image(box_max), color=blue, thickness=5)
            
            loc = self.point_to_image(box_min)
            loc = (loc[0], loc[1] + 30)
            cv2.putText(self.image, str(room_id), loc,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=3)

            self.room_meta.append({
                'room_idx': i,
                'room_id': room_id,
                'bbox_min': box_min,
                'bbox_max': box_max,
                'bbox_min_all': box_min_all,
                'bbox_max_all': box_max_all,
                'uids': uids
            })

    def draw_objects(self):
        for i in range(len(self.names)):
            x_list, y_list = [], []

            for j in range(8):
                temp = self.point_to_image(self.bboxes[i][j])
                x_list.append(temp[0])
                y_list.append(temp[1])

            x_min, y_min = self.point_to_image(self.bbox_mins[i])
            x_max, y_max = self.point_to_image(self.bbox_maxs[i])

            color = self.colors[i % 37]
            color = (int(color[0]), int(color[1]), int(color[2]))

            if self.names[i][:4] == 'Wall':
                cv2.rectangle(self.image, (x_min, y_min),
                              (x_max, y_max), (255, 255, 0), -1)
            elif self.names[i][:5] == 'Floor':
                pass
            else:
                for j in range(4):
                    cv2.line(self.image, (x_list[j], y_list[j]), 
                             (x_list[(j+1)%4], y_list[(j+1)%4]), color, 2)

                for j in range(4):
                    cv2.line(self.image, (x_list[j+4], y_list[j+4]), 
                             (x_list[(j+1)%4 + 4], y_list[(j+1)%4 + 4]), color, 2)

                for j in range(4):
                    cv2.line(self.image, (x_list[j], y_list[j]), 
                             (x_list[j+4], y_list[j+4]), color, 2)

                cv2.putText(self.image, self.names[i], (x_list[0], y_list[0] + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, thickness=2)

    def point_to_image(self, point_3d):
        """ Args:
                point_3d: raw float 3D point [x, y, z]
        """
        x = int(point_3d[0] * self.scale - self.scene_min[0] * self.scale + self.margin)
        y = self.height - int(point_3d[1] * self.scale - self.scene_min[1] * self.scale + self.margin)
        return (x, y)

    def save(self, path):
        cv2.imwrite(path, self.image)

    def save_room_meta(self, path):
        with open(path, 'w') as f:
            scene_id = os.path.basename(self.json_path)
            scene_id = scene_id.split('.')[0]

            room_meta = self.get_room_meta()
            for room in room_meta:
                for key in room:
                    if key == 'room_id':
                        continue
                    room[key] = room[key].tolist()

            json_dict = {
                'scene_id': scene_id,
                'room_metadata': room_meta,
            }
            json.dump(json_dict, f, indent=2)

    def drawgroups_and_save(self, path):
        self.draw_objects()
        self.draw_room_bbox()
        self.draw_coords()
        self.save(path)


if __name__ == '__main__':
    output_dir = '/data/bhuai/front3d_blender/layout'
    os.makedirs(output_dir, exist_ok=True)
    json_dir = '/data/bhuai/3D-FRONT'

    existed = os.listdir('/data/bhuai/front3d_ngp')
    scene_ids = [int(x.split('_')[1]) for x in existed]
    scene_ids = set(scene_ids)
    failed_scenes = []

    for f in os.listdir(output_dir):
        scene_id = int(f.split('_')[1].split('.')[0])
        scene_ids.add(scene_id)

    for i, scene in enumerate(sorted(os.listdir(json_dir))):
        if i in scene_ids:
            continue

        try:
            scene_json = os.path.join(json_dir, scene)
            drawer = FloorPlan(scene_json)
            drawer.drawgroups_and_save(os.path.join(output_dir, f'3dfront_{i:04d}.jpg'))
            drawer.save_room_meta(os.path.join(output_dir, f'3dfront_{i:04d}.json'))
        except:
            print(f'Failed to process {scene}')
            failed_scenes.append(scene)

    print('Failed scenes: ')
    print(failed_scenes)
