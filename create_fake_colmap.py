import os
import shutil
import numpy as np

def rotmat2q(R):
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])

src_dir = "views"
img_dir = "images"

if os.path.exists(img_dir):
    shutil.rmtree(img_dir)
os.makedirs(img_dir)
imgs = sorted([f for f in os.listdir(src_dir) if f.endswith('.png')])
for name in imgs:
    shutil.copy(os.path.join(src_dir, name), os.path.join(img_dir, name))

H, W = 512, 512
fov = 50
f = W / (2 * np.tan(np.deg2rad(fov / 2)))
K = np.array([[f, 0, W/2], [0, f, H/2], [0, 0, 1]])

os.makedirs("sparse/0", exist_ok=True)

radius = 2.0
cameras = []
for i, name in enumerate(imgs):
    azimuth = np.deg2rad(i * 360 / len(imgs))
    cx = radius * np.cos(azimuth)
    cy = 0.0
    cz = radius * np.sin(azimuth)
    forward = -np.array([cx, cy, cz]) / radius
    right = np.cross(np.array([0, 1, 0]), forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    R_w2c = np.stack([right, up, forward], axis=0)
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ np.array([cx, cy, cz])
    q = rotmat2q(R_c2w)
    cameras.append({'id': i+1, 'name': name, 'R_c2w': R_c2w, 't_c2w': t_c2w, 'q': q})

np.random.seed(0)
n_points = 3000
pts = np.random.randn(n_points, 3)
pts /= np.linalg.norm(pts, axis=1, keepdims=True)
pts *= np.random.uniform(0.1, 0.6, size=(n_points, 1))
colors = np.random.randint(0, 255, size=(n_points, 3))
point_tracks = [[] for _ in range(n_points)]
img_features = []

for cam in cameras:
    features = []
    R_w2c = cam['R_c2w'].T
    t_w2c = -R_w2c @ cam['t_c2w']
    for pid in range(n_points):
        P_c = R_w2c @ pts[pid] + t_w2c
        if P_c[2] <= 0.1:
            continue
        p = K @ P_c
        u, v = p[0]/p[2], p[1]/p[2]
        if 0 < u < W and 0 < v < H:
            features.append((u, v, pid+1))
            point_tracks[pid].append((cam['id'], len(features)-1))
    img_features.append(features)

with open("sparse/0/cameras.txt", "w") as fcam:
    fcam.write("# Camera list with one line of data per camera:\n")
    fcam.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    fcam.write(f"1 PINHOLE {W} {H} {f:.6f} {f:.6f} {W/2:.6f} {H/2:.6f}\n")

with open("sparse/0/images.txt", "w") as fimg:
    fimg.write("# Image list with two lines of data per image:\n")
    fimg.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    for i, cam in enumerate(cameras):
        q, t = cam['q'], cam['t_c2w']
        fimg.write(f"{cam['id']} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} 1 {cam['name']}\n")
        feats = img_features[i]
        if len(feats) == 0:
            fimg.write("0 0 -1\n")
        else:
            fimg.write(" ".join([f"{x:.6f} {y:.6f} {pid}" for x, y, pid in feats]) + "\n")

with open("sparse/0/points3D.txt", "w") as fpt:
    fpt.write("# 3D point list with one line of data per point:\n")
    fpt.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    for i in range(n_points):
        track = " ".join([f"{img_id} {pt_idx}" for img_id, pt_idx in point_tracks[i]])
        fpt.write(f"{i+1} {pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} {colors[i,0]} {colors[i,1]} {colors[i,2]} 1.0 {track}\n")

print(f"Done: {len(imgs)} images, {n_points} points")
