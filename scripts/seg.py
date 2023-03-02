from pysdf import SDF
import numpy as np
import json
import bpy
import bmesh
import mathutils
from tqdm import tqdm
from utils import poly2obb_3d


def get_sdf(mesh, xform):
    bm = bmesh.new()
    bm.from_mesh(mesh)

    mat = mathutils.Matrix(xform)
    bmesh.ops.transform(bm, matrix=mat, verts=bm.verts)

    vertices = np.array([v.co for v in bm.verts])
    faces = np.array([[v.index for v in f.verts] for f in bm.faces])

    sdf = SDF(vertices, faces)

    bm.free()
    return sdf


def build_segmentation_map(room_objs, room_bbox, max_res, res=None):
    """Builds a segmentation map of the room.

    Args:
        room_objs (list): A list of objects in the room.
        room_bbox (list): The bounding box of the room. (min, max)
        max_res (int): The max resolution of the segmentation map.

    Returns:
        (np.array, np.array, dict): A segmentation map of the room,
            resolution of the map, and a mapping from instance name
            to segmentation id.
    """

    # print(f'room bbox: {room_bbox}')

    diag = np.array(room_bbox[1]) - np.array(room_bbox[0])
    if res is None:
        res = diag / diag.max() * max_res
        res = np.floor(res).astype(np.int32)

    instance_map = np.zeros(res, dtype=np.uint8)     # instance id overflow?

    for obj in tqdm(room_objs):
        id = obj.get_cp('instance_id')
        bbox = obj.get_bound_box()
        aabb = np.array([np.min(bbox, axis=0), np.max(bbox, axis=0)])
        # print(f'{obj.get_name()} bbox: {aabb}')

        xs = np.linspace(room_bbox[0][0], room_bbox[1][0], res[0])
        ys = np.linspace(room_bbox[0][1], room_bbox[1][1], res[1])
        zs = np.linspace(room_bbox[0][2], room_bbox[1][2], res[2])

        x_start = np.searchsorted(xs, aabb[0][0])
        x_end = np.searchsorted(xs, aabb[1][0])
        y_start = np.searchsorted(ys, aabb[0][1])
        y_end = np.searchsorted(ys, aabb[1][1])
        z_start = np.searchsorted(zs, aabb[0][2])
        z_end = np.searchsorted(zs, aabb[1][2])

        mesh = obj.get_mesh()
        sdf = get_sdf(mesh, obj.get_local2world_mat())
        for i in range(x_start, x_end+1):
            for j in range(y_start, y_end+1):
                for k in range(z_start, z_end+1):
                    point = np.array([xs[i], ys[j], zs[k]])
                    if not sdf.contains(point):
                        continue

                    instance_map[i, j, k] = id

    return instance_map, res


def build_metadata(id_map, instances, room_bbox, model_info_path):
    """Builds metadata of the room.

    Args:
        id_map (dict): A mapping from instance name to segmentation id.
        room_obj_dict (dict): Dictionary of objects in the room.

    Returns:
        (dict): Dictionary of the room metadata.
    """

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    uid2info = {info['model_id']: info for info in model_info}

    metadata = {}
    metadata['scene_bbox'] = room_bbox.flatten().tolist()

    metadata['instances'] = []
    for instance in instances:
        volume = np.prod(instance['aabb'][1] - instance['aabb'][0])
        name = instance['name']

        if volume < 1e-6:
            print(f'Warning: {name} has volume {volume} and is ignored.')
            continue

        obj_data = {
            'name': name,
            'id': id_map[name],
            'aabb': instance['aabb'].flatten().tolist(),
            'obb': poly2obb_3d(instance['obb_corners']).tolist(),
            'uid': instance['uid'],
            'jid': instance['jid'],
            'super-category': uid2info[instance['jid']]['super-category'],
            'category': uid2info[instance['jid']]['category'],
        }

        metadata['instances'].append(obj_data)

    return metadata
