import os
import numpy as np
from plyfile import PlyData, PlyElement

def load_ply(path):
    plydata = PlyData.read(path)
    vertex = plydata['vertex'].data
    return vertex

def save_ply(vertex, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(path)

plys = [
    ('/root/autodl-tmp/datasets/garden/output/point_cloud/iteration_7000/point_cloud.ply', 1.0, (0, 0, 0)),
    ('/root/autodl-tmp/object_a/output/point_cloud/iteration_7000/point_cloud.ply', 0.1, (0, 0.5, 0)),
    ('/root/autodl-tmp/object_b/output/point_cloud/iteration_7000/point_cloud.ply', 0.1, (-1.5, 0.5, 0)),
    ('/root/autodl-tmp/object_c/output/point_cloud/iteration_7000/point_cloud.ply', 0.1, (1.5, 0.5, 0)),
]

vertices = []
for path, scale, translate in plys:
    v = load_ply(path)
    v = np.array(v, copy=True)
    v['x'] = v['x'] * scale + translate[0]
    v['y'] = v['y'] * scale + translate[1]
    v['z'] = v['z'] * scale + translate[2]
    if scale != 1.0:
        for i in range(3):
            v[f'scale_{i}'] = v[f'scale_{i}'] + np.log(scale)
    vertices.append(v)

merged = np.concatenate(vertices)
os.makedirs('/root/autodl-tmp/merge/point_cloud/iteration_7000', exist_ok=True)
save_ply(merged, '/root/autodl-tmp/merge/point_cloud/iteration_7000/point_cloud.ply')
print(f"Merged {len(vertices)} models, total points: {len(merged)}")
