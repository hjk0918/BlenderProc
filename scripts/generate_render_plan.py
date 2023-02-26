import json
import os


if __name__ == '__main__':
    room_types = ['bedroom', 'living', 'dining', 'kidsroom']

    layout_dir = '/data/bhuai/front3d_blender/layout'
    layout_list = os.listdir(layout_dir)
    layout_list = [x for x in layout_list if x.endswith('.json')]

    layout_list.sort()

    render_plan = {}
    for l in layout_list:
        layout_path = os.path.join(layout_dir, l)
        with open(layout_path, 'r') as f:
            layout = json.load(f)

        scene_name = l.split('.')[0]
        room_list = []

        rooms = layout['room_metadata']
        for room in rooms:
            room_id = room['room_id']
            
            is_used = False
            for type in room_types:
                if type in room_id.lower():
                    is_used = True
                    break

            bbox_min = room['bbox_min']
            bbox_max = room['bbox_max']

            if (max(bbox_max) > 10 or min(bbox_min) < -10) and is_used:
                print(f'{scene_name} {room_id} is too large: {bbox_min} {bbox_max}, skip')
                is_used = False

            if is_used:
                room_list.append(room_id)

        if len(room_list) > 0:
            render_plan[scene_name] = {
                'scene_id': layout['scene_id'],
                'rooms': room_list
            }

    with open('render_plan.json', 'w') as f:
        json.dump(render_plan, f, indent=2)
