import numpy as np

IMG_WIDTH = 640
IMG_HEIGHT = 480

K = np.array([
            [400, 0, 320],
            [0, 400, 240],
            [0, 0, 1]
            ])

CAMERA_LOCS = {0: [[0, 2.5], [1.4, -0.8]], 
               1: [[-1.5, 2]],#, [0, 1.5], [-3, -2]], 
               2: [[1, 0], [0, 0]],
               3: [[2, 0], [-1.5, 2.5]],
               4: [[2, 1], [2.5, -1]]}

# OBJ_BAN_LIST is a keyword ban list for all the rooms 
OBJ_BAN_LIST = ['Baseboard', 'Pocket', 'Floor', 'SlabSide.', 'WallInner', 'Front', 
                'WallTop', 'WallBottom', 'Ceiling.', 'FeatureWall', 'LightBand']

"""
<room config template>

0: {'bbox': [[-4.9, -0.7], [-0.7, 3]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': []
                        }

"""
ROOM_CONFIG = { 0: {0: {'bbox': [[-3.5, -3.9], [-0.8, -1]], 
                        },
                    1: {'bbox': [[0.5, -2.2], [4.3, 0.9]], 
                        'fullname_ban_list': ['lighting', 'lighting.002', 'lighting.003', 'lighting.004', 'lighting.005']
                        }
                    },
                1: {0: {'bbox': [[-4.9, -0.7], [-0.7, 3]], 
                        'keyword_ban_list': ['CustomizedFeatureWall', 'LightBand'],
                        'fullname_ban_list': ['sofa.003', 'sofa', 'media unit']
                        }
                    },
                2: {0: {'bbox': [[0.75, -0.8], [3.9, 2.4]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': []
                        }
                    },
                3: {0: {'center': (3, 2.2), 'a': 1.5, 'b': 2.2, 'num_cam': 16, 'bbox': [[0.8, -1.0], [5, 5]], 'corners': [[1.4, 0], [4.5, 4.3]],
                        'keyword_ban_list': ['Nightstand.001', 'Nightstand.003', 'Ceiling Lamp'],
                        'fullname_ban_list': []
                        },
                    1: {'center': (3.3, -4.8), 'a': 1.3, 'b': 2, 'num_cam': 8, 'bbox': [[1.3, -7.35], [5.0, -2.5]]},
                    2: {'center': (-0.4, -5), 'a': 1.5, 'b': 2.1, 'num_cam': 16, 'bbox': [[-2.2, -7.35], [1.3, -2.5]]}},
                4: {0: {'bbox': [[1.9, -3.9], [6.6, -0.5]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': []
                        },
                    },
                5: {0: {'bbox': [[0.6, -2.2], [4.3, 0.8]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': ['bed', 'bed.002','bed.003', 'bed.005', 'bed.006', 'bed.007','lighting.003']
                        },
                    1: {'bbox': [[-3.5, -4], [-0.8, -0.9]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': ['lighting.002']
                        },
                    2: {'bbox': [[1.1, -5.6], [5.9, -2.2]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': []
                        }
                    },
                6: {0: {'bbox': [[-6.1, -4.5], [-1.2, 2.1]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': []
                        }
                    },
                25:{0: {'bbox': [[0.2, -3.1], [3.6, -0.2]], 
                        'keyword_ban_list': [],
                        'fullname_ban_list': []
                        }}
                }   
