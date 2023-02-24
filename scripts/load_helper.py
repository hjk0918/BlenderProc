import numpy as np
import os

import blenderproc as bproc
from blenderproc.python.types.MeshObjectUtility import MeshObject
import re
import bpy
import json

LAYOUT_DIR = '/data/jhuangce/3D-FRONT'
TEXTURE_DIR = '/data/jhuangce/3D-FRONT-texture'
MODEL_DIR = '/data/jhuangce/3D-FUTURE-model'

RENDER_TEMP_DIR = './FRONT3D_render/temp'
SCENE_LIST = []


def get_scene_rot_bbox_meta(loaded_objects):
    names = []
    bboxes = []
    uids = []
    room_ids = []
    
    for i in range(len(loaded_objects)):
        object = loaded_objects[i]
        name = object.get_name()
        bbox = object.get_bound_box()
        uid = object.get_cp('uid')

        if not np.isfinite(bbox).all():
            continue

        if object.has_cp('room_id'):
            room_id = object.get_cp('room_id')
        else:
            room_id = -1

        names.append(name)
        bboxes.append(bbox)
        uids.append(uid)
        room_ids.append(room_id)

    return names, bboxes, uids, room_ids


def add_texture(obj: MeshObject, tex_path):
    """ Add a texture to an object. """
    obj.clear_materials()
    mat = obj.new_material('my_material')
    bsdf = mat.nodes["Principled BSDF"]
    texImage = mat.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(tex_path)
    mat.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])


# TODO: read config file
def load_scene_objects(json_path):
    mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
    mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)
    
    loaded_objects = bproc.loader.load_front3d(
        json_path=json_path,
        future_model_path=MODEL_DIR,
        front_3D_texture_path=TEXTURE_DIR,
        label_mapping=mapping,
        ceiling_light_strength=1,
        lamp_light_strength=30
    )

    # add texture to wall and floor. Otherwise they will be white.
    for obj in loaded_objects:
        # skip objects that already have texture
        if not obj.has_cp('has_3D_future_texture') or obj.get_cp('has_3D_future_texture'):
            continue

        name = obj.get_name()
        if 'wall' in name.lower():
            add_texture(obj, TEXTURE_DIR+"/1b57700d-f41b-4ac7-a31a-870544c3d608/texture.png")
        elif 'floor' in name.lower():
            add_texture(obj, TEXTURE_DIR+"/0b48b46d-4f0b-418d-bde6-30ca302288e6/texture.png")

    return loaded_objects
