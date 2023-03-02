# python cli.py run ./scripts/utils.py 
import blenderproc as bproc

"""
    Example commands:

        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode plan
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode overview 
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -ppo 10 -gd 0.15
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -ppo 0 -gp 5
        python cli.py run ./scripts/render.py --gpu 3 -s 0 -r 0 --mode render -nc

"""

from random import shuffle
import shutil
import sys
# sys.path.append('/home/jhuangce/miniconda3/lib/python3.9/site-packages')
# sys.path.append('/home/yliugu/BlenderProc/scripts')
sys.path.append('./scripts')
import cv2
import os
from os.path import join
import numpy as np

import imageio
import sys
# sys.path.append('/data/jhuangce/BlenderProc/scripts')
# sys.path.append('/data2/jhuangce/BlenderProc/scripts')
from floor_plan import *
from load_helper import *
from render_configs import *
from utils import *
from bbox_proj import get_aabb_coords, project_aabb_to_image, project_obb_to_image
import json
from typing import List
from os.path import join
from collections import defaultdict
import glob
import argparse
from mathutils import Vector, Matrix

import pandas as pd
from seg import build_segmentation_map, build_metadata


pi = np.pi
cos = np.cos
sin = np.sin
COMPUTE_DEVICE_TYPE = "CUDA"

RENDER_PLAN_PATH = '/data/bhuai/front3d_blender/render_plan.json'
MODEL_INFO_PATH = '/data/bhuai/3D-FUTURE-model/model_info.json'


def construct_scene_list():
    """ Construct a list of scenes and save to SCENE_LIST global variable. """
    scene_list = sorted([join(LAYOUT_DIR, name) for name in os.listdir(LAYOUT_DIR)])
    for scene_path in scene_list:
        SCENE_LIST.append(scene_path)
    print(f"SCENE_LIST is constructed. {len(SCENE_LIST)} scenes in total")


############################## poses generation ##################################

def normalize(x, axis=-1, order=2):
    l2 = np.linalg.norm(x, order, axis)
    l2 = np.expand_dims(l2, axis)
    l2[l2 == 0] = 1
    return x / l2,

def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.
    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """

    if at is None:
        at = np.zeros_like(camera_position)
    else:
        at = np.array(at)
    if up is None:
        up = np.zeros_like(camera_position)
        up[2] = -1
    else:
        up = np.array(up)
    
    z_axis = normalize(camera_position - at)[0]
    x_axis = normalize(np.cross(up, z_axis))[0]
    y_axis = normalize(np.cross(z_axis, x_axis))[0]

    R = np.concatenate([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R

def c2w_from_loc_and_at(cam_pos, at, up=(0, 0, 1)):
    """ Convert camera location and direction to camera2world matrix. """
    c2w = np.eye(4)
    cam_rot = look_at_rotation(cam_pos, at=at, up=up, inverse=False, cv=True)
    c2w[:3, 3], c2w[:3, :3] = cam_pos, cam_rot
    return c2w

def generate_four_corner_poses(room_bbox):
    """ Return a list of matrices of 4 corner views in the room. """
    bbox_xy = room_bbox[:, :2]
    corners = [[i+0.3 for i in bbox_xy[0]], [i-0.3 for i in bbox_xy[1]]]
    x1, y1, x2, y2 = corners[0][0], corners[0][1], corners[1][0], corners[1][1]
    at = [(x1+x2)/2, (y1+y2)/2, 1.2]
    locs = [[x1, y1, 2], [x1, y2, 2], [x2, y1, 2], [x2, y2, 2]]

    c2ws = [c2w_from_loc_and_at(pos, at) for pos in locs]
    
    return c2ws

def pos_in_bbox(pos, bbox):
    """
    Check if a point is inside a bounding box.
    Input:
        pos: 3 x 1
        bbox: 2 x 3
    Output:
        True or False
    """
    return  pos[0] >= bbox[0][0] and pos[0] <= bbox[1][0] and \
            pos[1] >= bbox[0][1] and pos[1] <= bbox[1][1] and \
            pos[2] >= bbox[0][2] and pos[2] <= bbox[1][2]

def check_pos_valid(pos, instances, room_bbox):
    """ Check if the position is in the room, not too close to walls and not conflicting with other objects. """
    room_bbox_small = [
        [item+0.5 for item in room_bbox[0]], 
        [room_bbox[1][0]-0.5, room_bbox[1][1]-0.5, room_bbox[1][2]-0.8]
    ] # ceiling is lower

    if not pos_in_bbox(pos, room_bbox_small):
        return False
    for obj_dict in instances:
        obj_bbox = obj_dict['aabb']
        if pos_in_bbox(pos, obj_bbox):
            return False

    return True

def generate_room_poses(instances, room_bbox, num_poses_per_object, max_global_pos, global_density):
    """ Return a list of poses including global poses and close-up poses for each object."""

    poses = []
    num_closeup, num_global = 0, 0
    h_global = 1.2

    # close-up poses for each object.
    if num_poses_per_object>0:
        for obj_dict in instances:
            obj_bbox = np.array(obj_dict['aabb'])
            cent = np.mean(obj_bbox, axis=0)
            rad = np.linalg.norm(obj_bbox[1]-obj_bbox[0])/2 * 1.7 # how close the camera is to the object
            if np.max(obj_bbox[1]-obj_bbox[0])<1:
                rad *= 1.2 # handle small objects

            positions = []
            n_hori_sects = 30
            n_vert_sects = 10
            theta_bound = [0, 2*pi]
            phi_bound = [-pi/4, pi/4]
            theta_sect = (theta_bound[1] - theta_bound[0]) / n_hori_sects
            phi_sect = (phi_bound[1] - phi_bound[0]) / n_vert_sects
            for i_vert_sect in range(n_vert_sects):
                for i_hori_sect in range(n_hori_sects):
                    theta_a = theta_bound[0] + i_hori_sect * theta_sect
                    theta_b = theta_a + theta_sect
                    phi_a = phi_bound[0] + i_vert_sect * phi_sect
                    phi_b = phi_a + phi_sect
                    theta = np.random.uniform(theta_a, theta_b)
                    phi = np.random.uniform(phi_a, phi_b)
                    pos = [cos(phi)*cos(theta), cos(phi)*sin(theta), sin(phi)]
                    positions.append(pos)
            positions = np.array(positions)
            positions = positions * rad + cent

            positions = [pos for pos in positions if check_pos_valid(pos, instances, room_bbox)]
            shuffle(positions)
            if len(positions) > num_poses_per_object:
                positions = positions[:num_poses_per_object]

            poses.extend([c2w_from_loc_and_at(pos, cent) for pos in positions])

            num_closeup = len(positions)

    # global poses
    if max_global_pos>0:
        bbox = room_bbox
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1]
        rm_cent = np.array([(x1+x2)/2, (y1+y2)/2, h_global])

        # flower model
        rad_bound = [0.3, 5]
        rad_intv = global_density
        theta_bound = [0, 2*pi]
        theta_sects = 20
        theta_intv = (theta_bound[1] - theta_bound[0]) / theta_sects
        h_bound = [0.8, 2.0]

        positions = []
        theta = theta_bound[0]
        for i in range(theta_sects):
            rad = rad_bound[0]
            while rad < rad_bound[1]:
                h = np.random.uniform(h_bound[0], h_bound[1])
                pos = [rm_cent[0] + rad * cos(theta), rm_cent[1] + rad * sin(theta), h]
                if check_pos_valid(pos, instances, room_bbox):
                    positions.append(pos)
                rad += rad_intv
            theta += theta_intv
        positions = np.array(positions)
        np.random.shuffle(positions)

        if len(positions) > max_global_pos:
            positions = positions[:max_global_pos]

        poses.extend([c2w_from_loc_and_at(pos, [rm_cent[0], rm_cent[1], pos[2]]) for pos in positions])

        num_global = len(positions)
        

    return poses, num_closeup, num_global


#################################################################################

def get_scene_bbox(loaded_objects=None, scene_objs_dict=None):
    """ Return the bounding box of the scene. """
    bbox_mins = []
    bbox_maxs = []
    if loaded_objects!=None:
        for i, object in enumerate(loaded_objects):
            bbox = object.get_bound_box()
            bbox_mins.append(np.min(bbox, axis=0))
            bbox_maxs.append(np.max(bbox, axis=0))
            scene_min = np.min(bbox_mins, axis=0)
            scene_max = np.max(bbox_maxs, axis=0)
            return scene_min, scene_max
    elif scene_objs_dict!=None:
        return scene_objs_dict['bbox']
    else:
        raise ValueError('Either loaded_objects or scene_objs_dict should be provided.')
    

def get_room_bbox(scene_idx, room_idx, scene_objects=None, scene_objs_dict=None):
    """ Return the bounding box of the room. """
    # get global height
    scene_min, scene_max = get_scene_bbox(scene_objects, scene_objs_dict)
    room_config = ROOM_CONFIG[scene_idx][room_idx]
    # overwrite width and length with room config
    scene_min[:2] = room_config['bbox'][0]
    scene_max[:2] = room_config['bbox'][1]

    return [scene_min, scene_max]

def bbox_contained(bbox_a, bbox_b):
    """ Return whether the bbox_a is contained in bbox_b. """
    return bbox_a[0][0]>=bbox_b[0][0] and bbox_a[0][1]>=bbox_b[0][1] and bbox_a[0][2]>=bbox_b[0][2] and \
           bbox_a[1][0]<=bbox_b[1][0] and bbox_a[1][1]<=bbox_b[1][1] and bbox_a[1][2]<=bbox_b[1][2]

def get_room_objects(scene_objects, room_bbox, cleanup=False):
    """ Return the objects within the room bbox. Cleanup unecessary objects. """
    objects = []

    for object in scene_objects:
        obj_bbox = object.get_bound_box()
        aabb = np.array([np.min(obj_bbox, axis=0), np.max(obj_bbox, axis=0)])
        if bbox_contained(aabb, room_bbox):
            objects.append(object)

    return objects

def merge_bbox(scene_idx, room_idx, room_bbox_meta):
    """ Merge the bounding box of the room. """
    if 'merge_list' in ROOM_CONFIG[scene_idx][room_idx]:
        merge_dict = ROOM_CONFIG[scene_idx][room_idx]['merge_list']
        for label, merge_items in merge_dict.items():
            result_room_bbox_meta, merge_mins, merge_maxs = [], [], []
            for obj in room_bbox_meta:
                if obj[0] in merge_items:
                    merge_mins.append(obj[1][0])
                    merge_maxs.append(obj[1][1])
                else:
                    result_room_bbox_meta.append(obj)
            if len(merge_mins) > 0:
                result_room_bbox_meta.append((label, [np.min(np.array(merge_mins), axis=0), np.max(np.array(merge_maxs), axis=0)]))
            room_bbox_meta = result_room_bbox_meta
    return room_bbox_meta

def merge_bbox_in_dict(scene_idx, room_idx, room_objs_dict):
    """ Merge the bounding box of the room. Operate on the object dictionary with obb """
    if 'merge_list' in ROOM_CONFIG[scene_idx][room_idx]:
        merge_dict = ROOM_CONFIG[scene_idx][room_idx]['merge_list']
        objects = room_objs_dict['objects']
        for merged_label, merge_items in merge_dict.items():
            # select objs to be merged
            result_objects = [obj for obj in objects if obj['name'] not in merge_items]
            objs_to_be_merged = [obj for obj in objects if obj['name'] in merge_items]

            # find the largest object
            largest_obj = None
            largest_vol = 0
            for obj in objs_to_be_merged:
                if obj['volume'] > largest_vol:
                    largest_vol = obj['volume']
                    largest_obj = obj
            
            # extend the largest bbox to include all the other bbox
            local2world = Matrix(largest_obj['l2w'])
            local_maxs, local_mins = np.max(largest_obj['coords_local'], axis=0), np.min(largest_obj['coords_local'], axis=0)
            local_cent = (local_maxs + local_mins) / 2
            global_cent = local2world @ Vector(local_cent)
            h_diag = (local_maxs - local_mins) / 2
            local_vecs = np.array([[h_diag[0], 0, 0], [0, h_diag[1], 0], [0, 0, h_diag[2]]]) + local_cent  # (3, 3)
            global_vecs = [(local2world @ Vector(vec) - local2world @ Vector(local_cent)).normalized() for vec in local_vecs] # (3, 3)
            global_norms = [vec for vec in global_vecs] # (3, 3)
            local_offsets = np.array([-h_diag, h_diag]) # [[x-, y-, z-], [x+, y+, z+]]

            for obj in objs_to_be_merged:
                update = [[0, 0, 0], [0, 0, 0]]
                for point in obj['coords']:
                    for i in range(3):
                        offset = (Vector(point) - global_cent) @ global_norms[i]
                        if offset < local_offsets[0][i]:
                            local_offsets[0][i] = offset
                            update[0][i] = 1
                        elif offset > local_offsets[1][i]:
                            local_offsets[1][i] = offset
                            update[1][i] = 1
            
            # TODO: update: coords, aabb, volume, coords_local
            # TODO: Compute real aabb by creating parent object
            merged_local_mins, merged_local_maxs = local_offsets + local_cent
            merged_coords_local = get_aabb_coords(np.concatenate([merged_local_mins, merged_local_maxs], axis=0))[:, :3]
            merged_coords = np.array([local2world @ Vector(cord) for cord in merged_coords_local])
            merged_aabb_mins, merged_aabb_maxs = np.min(merged_coords, axis=0), np.max(merged_coords, axis=0)
            merged_aabb = np.array([merged_aabb_mins, merged_aabb_maxs])
            merged_diag_local = merged_local_maxs - merged_local_mins
            merged_volume = merged_diag_local[0] * merged_diag_local[1] * merged_diag_local[2]


            merged_object = {'name': merged_label,
                             'coords': merged_coords,
                             'aabb': merged_aabb,
                             'volume': merged_volume,
                             'l2w': largest_obj['l2w'],
                             'coords_local': merged_coords_local,}
            result_objects.append(merged_object)
            objects = result_objects

        room_objs_dict['objects'] = objects

    return room_objs_dict


def filter_objs(objects, room_idx):
    for obj in objects:
        obj.set_cp('is_filtered', False)

        # object without room_id is not furniture
        if not obj.has_cp('room_id'):
            obj.set_cp('is_filtered', True)
            continue

        # check whether the object is in the room
        if obj.get_cp('room_id') != room_idx:
            obj.set_cp('is_filtered', True)
            continue

        obj_name = obj.get_name()

        # check global OBJ_BAN_LIST
        for ban_word in OBJ_BAN_LIST:
            if ban_word in obj_name:
                obj.set_cp('is_filtered', True)
                continue


def get_instance_data(objects):
    '''
    Merge mesh of the same instance, compute new bounding box, and rename the object.
    '''
    uid2instances = defaultdict(list)
    uid2jid = {}

    for obj in objects:
        if not obj.has_cp('is_filtered'):
            raise ValueError('Object has no is_filtered attribute, call filter_objs() first.')

        if obj.get_cp('is_filtered'):
            continue

        uid = obj.get_cp('uid')
        jid = obj.get_cp('jid')
        instance_id = obj.get_cp('uid_instance_id')
        uid2instances[uid].append((instance_id, obj))
        uid2jid[uid] = jid

    instance_data = []
    for uid, instances in uid2instances.items():
        ids = [instance_id for instance_id, obj in instances]
        ids = set(ids)
        for instance_id in ids:
            instance_objs = [obj for id, obj in instances if id == instance_id]

            # merge mesh
            obb_corners, aabb = compute_objects_bbox(instance_objs)
            name = instance_objs[0].get_name().split('.')[0]
            name = f'{name}.{instance_id:03d}'

            data = {
                'name': name,
                'obb_corners': obb_corners,
                'aabb': aabb,
                'instance_id': instance_id,
                'uid': uid,
                'jid': uid2jid[uid],
                'objects': instance_objs,
            }

            instance_data.append(data)

    return instance_data


# For metadata and 2D/3D mask generation
def build_id_map(instances):
    id_map = {}
    for instance in instances:
        cur_id = len(id_map) + 1
        id_map[instance['name']] = cur_id
        for obj in instance['objects']:
            obj.set_cp('instance_name', instance['name'])
            obj.set_cp('instance_id', cur_id)
    
    return id_map


def render_poses(poses, temp_dir=RENDER_TEMP_DIR) -> List:
    """ Render a scene with a list of poses. 
        No room idx is needed because the poses can be anywhere in the room. """

    # add camera poses to render queue
    for cam2world_matrix in poses:
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    # render
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200, 
        transmission_bounces=200, transparent_max_bounces=200)
    bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)
    data = bproc.renderer.render(output_dir=temp_dir)
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in data['colors']]

    return imgs

##################################### save to dataset #####################################
def get_ngp_type_boxes(instances, bbox_type):
    """ Return a list of bbox in instant-ngp format. """
    bounding_boxes = []
    for i, obj_dict in enumerate(instances):
        if bbox_type == 'aabb':
            obj_aabb = obj_dict['aabb']
            obj_bbox_ngp = {
                "extents": (obj_aabb[1]-obj_aabb[0]).tolist(),
                "orientation": np.eye(3).tolist(),
                "position": ((obj_aabb[0]+obj_aabb[1])/2.0).tolist(),
            }
            bounding_boxes.append(obj_bbox_ngp)
        elif bbox_type == 'obb':
            obj_coords = obj_dict['obb_corners']
            # TODO: 8 point to [x, y, z, w, l, h, theta]
            np.set_printoptions(precision=2)
            obb = poly2obb_3d(obj_coords)
            extents, orientation, position = obb2ngp(obb)

            if obj_dict['name'] == 'chair':
                print('x y z w l h theta', obb)
                print('extents', extents)
                print('orientation', orientation)
                print('position', position)
            obj_bbox_ngp = {
                "extents": extents,
                "orientation": orientation,
                "position": position,
            }
            bounding_boxes.append(obj_bbox_ngp)
    return bounding_boxes

def save_in_ngp_format(imgs, poses, intrinsic, instances, room_bbox, bbox_type, dst_dir):
    """ Save images and poses to ngp format dataset. """
    print('Save in instant-ngp format...')
    train_dir = join(dst_dir, 'train')
    imgdir = join(dst_dir, 'train', 'images')

    if os.path.isdir(imgdir) and len(os.listdir(imgdir))>0:
        input("Warning: The existing images will be overwritten. Press enter to continue...")
        shutil.rmtree(imgdir)
    os.makedirs(imgdir, exist_ok=True)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    angle_x = 2*np.arctan(cx/fx)
    angle_y = 2*np.arctan(cy/fy)

    scale = 1.5 / np.max(room_bbox[1] - room_bbox[0])
    cent_after_scale = scale * (room_bbox[0] + room_bbox[1])/2.0
    offset = np.array([0.5, 0.5, 0.5]) - cent_after_scale

    out = {
        "camera_angle_x": float(angle_x),
        "camera_angle_y": float(angle_y),
        "fl_x": float(fx),
        "fl_y": float(fy),
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "cx": float(cx),
        "cy": float(cy),
        "w": int(IMG_WIDTH),
        "h": int(IMG_HEIGHT),
        "aabb_scale": 2,
        "scale": float(scale),
        "offset": offset.tolist(),
        "room_bbox": room_bbox.tolist(),
        "num_room_objects": len(instances),
        "frames": [],
        "bounding_boxes": []
    }
    
    for i, pose in enumerate(poses):
        frame = {
            "file_path": join('images/{:04d}.jpg'.format(i)),
            "transform_matrix": pose.tolist()
        }
        out['frames'].append(frame)
    
    out['bounding_boxes'] = get_ngp_type_boxes(instances, bbox_type)

    # out['is_merge_bbox'] = 'No'
    
    with open(join(train_dir, 'transforms.json'), 'w') as f:
        json.dump(out, f, indent=2)
    
    if imgs == None: # support late rendering
        imgs = render_poses(poses, imgdir)
    
    for i, img in enumerate(imgs):
        cv2.imwrite(join(imgdir, '{:04d}.jpg'.format(i)), img)



###########################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--scene_idx', type=int, required=True)
    parser.add_argument('-r', '--room_idx', type=int, required=True)
    parser.add_argument('--mode', type=str, choices=['plan', 'overview', 'render', 'bbox', 'seg', 'depth'], 
                        help="plan: Generate the floor plan of the scene. \
                              overview:Generate 4 corner overviews with bbox projected. \
                              render: Render images in the scene. \
                              bbox: Overwrite bboxes by regenerating transforms.json."
                              "\nseg: Create 3D semantic/instance segmentation map.")
    parser.add_argument('-ppo', '--pos_per_obj', type=int, default=15, help='Number of close-up poses for each object.')
    parser.add_argument('-gp', '--max_global_pos', type=int, default=150, help='Max number of global poses.')
    parser.add_argument('-gd', '--global_density', type=float, default=0.15, help='The radius interval of global poses. Smaller global_density -> more global views')
    parser.add_argument('-nc', '--no_check', action='store_true', default=False, help='Do not check the poses. Render directly.')
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--relabel', action='store_true', help='Relabel the objects in the scene by rewriting transforms.json.')
    parser.add_argument('--bbox_type', type=str, default="aabb", choices=['aabb', 'obb'], help='Output aabb or obb')
    parser.add_argument('--render_root', type=str, default='./FRONT3D_render', help='Output directory. If not specified, use the default directory.')

    parser.add_argument('--seg_res', type=int, default=256, help='The max grid resolution for 3D segmentation map.')
    parser.add_argument('--pose_dir', type=str, default='', 
                        help='The directory containing the poses (transforms.json) for 2D mask rendering.')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu



    return parser.parse_args()


def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.scene_idx < 0 or args.scene_idx > 6812:
        raise ValueError(f"{args.scene_idx} is not a valid scene_idx. "
            "Should provide a scene_idx between 0 and 6812 inclusively")

    with open(RENDER_PLAN_PATH) as f:
        render_plan = json.load(f)

    room_id = render_plan[f'3dfront_{args.scene_idx:04d}']['rooms'][args.room_idx]

    dst_dir = join(args.render_root, '3dfront_{:04d}_{:02}'.format(args.scene_idx, args.room_idx))
    os.makedirs(dst_dir, exist_ok=True)

    construct_scene_list()

    bproc.init(compute_device='cuda:0', compute_device_type=COMPUTE_DEVICE_TYPE)
    bproc.renderer.set_noise_threshold(0.1)
    bproc.renderer.set_max_amount_of_samples(256)
    
    scene_objects = load_scene_objects(SCENE_LIST[args.scene_idx])
    # scene_objs_dict = build_and_save_scene_cache(cache_dir, scene_objects)

    floor_plan = FloorPlan(SCENE_LIST[args.scene_idx], loaded_objects=scene_objects)
    room_meta_all = floor_plan.get_room_meta()

    if args.mode == 'plan':
        os.makedirs(os.path.join(dst_dir, 'overview'), exist_ok=True)
        floor_plan.drawgroups_and_save(os.path.join(dst_dir, 'overview'))
        return

    room_meta = None
    for r in room_meta_all:
        if r['room_id'] == room_id:
            room_bbox = np.array([r['bbox_min'], r['bbox_max']])
            room_meta = r
            break

    if room_meta is None:
        raise ValueError(f"Room {room_id} not found in scene {args.scene_idx:04d}")

    # room_bbox = get_room_bbox(args.scene_idx, args.room_idx, scene_objs_dict=scene_objs_dict)
    # room_objs_dict = get_room_objs_dict(room_bbox, scene_objs_dict)
    # room_objs_dict = filter_objs_in_dict(args.scene_idx, args.room_idx, room_objs_dict)
    filter_objs(scene_objects, room_meta['room_idx'])
    instance_data = get_instance_data(scene_objects)

    if args.mode == 'overview':
        overview_dir = os.path.join(dst_dir, 'overview')
        os.makedirs(overview_dir, exist_ok=True)
        poses = generate_four_corner_poses(room_bbox)

        cache_dir = join(dst_dir, 'overview/raw')
        cached_img_paths = glob.glob(cache_dir+'/*')
        imgs = []
        if len(cached_img_paths) > 0 and True:
            # use cached overview images
            for img_path in sorted(cached_img_paths):
                imgs.append(cv2.imread(img_path))
        else:
            # render overview images
            imgs = render_poses(poses, overview_dir)
            os.makedirs(cache_dir, exist_ok=True)
            for i, img in enumerate(imgs):
                cv2.imwrite(join(cache_dir, f'raw_{i}.jpg'), img)

        # project aabb and obb to images
        labels, coords, aabb_codes, colors = [], [], [], []
        for obj in instance_data:
            labels.append(obj['name'])
            
            coord = np.array(obj['obb_corners'])
            coord = np.concatenate((coord, np.ones((len(coord), 1))), axis=1)
            coords.append(coord)

            aabb_codes.append(obj['aabb'])
            colors.append(random_color())

        aabb_codes = np.array(aabb_codes).reshape(-1, 6)
        
        for i, (img, pose) in enumerate(zip(imgs, poses)):
            img_aabb = project_aabb_to_image(img, K, np.linalg.inv(pose), aabb_codes, labels, colors)
            img_obb = project_obb_to_image(img, K, np.linalg.inv(pose), coords, labels, colors)
            cv2.imwrite(os.path.join(os.path.join(dst_dir, 'overview'), 'proj_aabb_{}.png'.format(i)), img_aabb)
            cv2.imwrite(os.path.join(os.path.join(dst_dir, 'overview'), 'proj_obb_{}.png'.format(i)), img_obb)
            
        for label in sorted(labels):
            print(label)
        print(f"{len(labels)} objects in total.\n")
    
    elif args.mode == 'render':
        poses, num_closeup, num_global = generate_room_poses(
            instance_data, room_bbox, 
            num_poses_per_object = args.pos_per_obj,
            max_global_pos = args.max_global_pos,
            global_density=args.global_density
        )
        
        if not args.no_check:
            print('Render for scene {}, room {}:'.format(args.scene_idx, args.room_idx))
            for obj_dict in instance_data:
                print(f"\t{obj_dict['aabb']}")
            print('Total poses: {}[global] + {}[closeup] x {}[object] = {} poses'.format(num_global, 
                args.pos_per_obj, len(instance_data), len(poses)))
            print('Estimated time: {} minutes'.format(len(poses)*25//60))
            input('Press Enter to continue...')

        save_in_ngp_format(None, poses, K, instance_data, room_bbox, args.bbox_type, dst_dir) # late rendering
        
    elif args.mode == 'bbox':
        json_path = os.path.join(dst_dir, 'train/transforms.json')
        json_path_result = os.path.join(dst_dir, 'train/transforms.json')

        with open(json_path, 'r') as f:
            meta = json.load(f)
        
        meta['bounding_boxes'] = get_ngp_type_boxes(instance_data, args.bbox_type)
        
        with open(json_path_result, 'w') as f:
            json.dump(meta, f, indent=2)

    elif args.mode == 'seg':
        id_map = build_id_map(instance_data)
        print('Number of objects in the room: ', len(instance_data))

        # data_path = os.path.join('/data/bhuai/3dfront_rpn_data/features_256',
        #                          f'3dfront_{args.scene_idx:04d}_{args.room_idx:02d}.npz')
        # if not os.path.exists(data_path):
        #     return
        # data = np.load(data_path)
        # res = data['resolution']
        # res = res[[2, 0, 1]]
        # res = res.astype(np.int32)

        # ins_map, res = build_segmentation_map(room_objs, room_bbox, args.seg_res, res)
        metadata = build_metadata(id_map, instance_data, room_bbox, MODEL_INFO_PATH)
        
        mask_dir = os.path.join(args.render_root, 'masks')
        os.makedirs(mask_dir, exist_ok=True)
        metadata_dir = os.path.join(args.render_root, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)

        scene_name = f'3dfront_{args.scene_idx:04d}_{args.room_idx:02d}'
        metadata['scene_name'] = scene_name
        metadata['scene_id'] = os.path.basename(SCENE_LIST[args.scene_idx]).split('.')[0]

        # np.save(os.path.join(mask_dir, scene_name + '.npy'), ins_map)
        with open(os.path.join(metadata_dir, scene_name + '.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        # Render 2D segmentation masks
        poses_file = os.path.join(args.pose_dir, scene_name, 'train', 'transforms.json')
        with open(poses_file) as f:
            data = json.load(f)
            for frame in data['frames']:
                pose = np.array(frame['transform_matrix'])
                bproc.camera.add_camera_pose(pose)
    
        bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)
        data = bproc.renderer.render_segmap(map_by=['instance', 'cp_instance_id'], 
                                            default_values={'cp_instance_id': 0})
        
        seg_dir = os.path.join(args.render_root, 'seg', scene_name)
        os.makedirs(seg_dir, exist_ok=True)
        bproc.writer.write_hdf5(seg_dir, data)

        # Render depth at the same time
        bproc.renderer.set_max_amount_of_samples(1)
        bproc.renderer.set_noise_threshold(0)
        bproc.renderer.set_denoiser(None)
        bproc.renderer.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
        bproc.renderer.enable_depth_output(activate_antialiasing=False)

        depth_data = bproc.renderer.render()
        
        depth_dir = os.path.join(args.render_root, 'depth', scene_name)
        os.makedirs(depth_dir, exist_ok=True)
        bproc.writer.write_hdf5(depth_dir, {'depth': depth_data['depth']})

    elif args.mode == 'depth':
        # Render depth images
        scene_name = f'3dfront_{args.scene_idx:04d}_{args.room_idx:02d}'
        poses_file = os.path.join(args.pose_dir, scene_name, 'train', 'transforms.json')
        with open(poses_file) as f:
            data = json.load(f)
            for frame in data['frames']:
                pose = np.array(frame['transform_matrix'])
                bproc.camera.add_camera_pose(pose)
    
        bproc.camera.set_intrinsics_from_K_matrix(K, IMG_WIDTH, IMG_HEIGHT)

        bproc.renderer.set_max_amount_of_samples(1)
        bproc.renderer.set_noise_threshold(0)
        bproc.renderer.set_denoiser(None)
        bproc.renderer.set_light_bounces(1, 0, 0, 1, 0, 8, 0)
        bproc.renderer.enable_depth_output(activate_antialiasing=False)

        data = bproc.renderer.render()
        
        depth_dir = os.path.join(args.render_root, 'depth', scene_name)
        os.makedirs(depth_dir, exist_ok=True)
        bproc.writer.write_hdf5(depth_dir, {'depth': data['depth']})


if __name__ == '__main__':
    main()
    print("Success.")
