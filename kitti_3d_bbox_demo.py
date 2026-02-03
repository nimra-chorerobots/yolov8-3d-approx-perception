import os
import glob
import time
import numpy as np
import cv2
import torch
import open3d as o3d

# =========================
# CONFIG (EDIT ONLY THIS)
# =========================
KITTI_ROOT = "/media/nimra/New Volume/Chorerobotics/codes/kitti"

IMG_DIR   = os.path.join(KITTI_ROOT, "data_object_image_2",   "training", "image_2")
LIDAR_DIR = os.path.join(KITTI_ROOT, "data_object_velodyne",  "training", "velodyne")
CALIB_DIR = os.path.join(KITTI_ROOT, "data_object_calib",     "training", "calib")
LABEL_DIR = os.path.join(KITTI_ROOT, "data_object_label_2",   "training", "label_2")

YOLO_MODEL = "yolov8n.pt"
MIDAS_MODEL = "MiDaS_small"

DRAW_GT_3D_ON_IMAGE = True
MAX_FRAMES = 200

# 🔥 ADDED: Playback speed control (IMPORTANT)
PLAYBACK_FPS = 1.0   # 0.5 = slow, 1.0 = normal, 2.0 = fast

# =========================
# HELPERS
# =========================
def read_calib(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k] = np.array([float(x) for x in v.split()], dtype=np.float32)

    P2 = data["P2"].reshape(3, 4)
    R0 = data.get("R0_rect", np.eye(3).reshape(-1)).reshape(3, 3)
    Tr = data["Tr_velo_to_cam"].reshape(3, 4)

    R0_4 = np.eye(4)
    R0_4[:3, :3] = R0

    Tr_4 = np.eye(4)
    Tr_4[:3, :4] = Tr

    return P2, R0_4, Tr_4

def parse_kitti_labels(label_path):
    objs = []
    if not os.path.exists(label_path):
        return objs

    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if p[0] == "DontCare":
                continue
            objs.append({
                "cls": p[0],
                "bbox2d": list(map(float, p[4:8])),
                "h": float(p[8]), "w": float(p[9]), "l": float(p[10]),
                "x": float(p[11]), "y": float(p[12]), "z": float(p[13]),
                "ry": float(p[14])
            })
    return objs

def corners_3d_in_camera(o):
    h,w,l = o["h"],o["w"],o["l"]
    x,y,z = o["x"],o["y"],o["z"]
    ry = o["ry"]

    x_c = [ l/2, l/2,-l/2,-l/2, l/2, l/2,-l/2,-l/2]
    y_c = [ 0,0,0,0,-h,-h,-h,-h]
    z_c = [ w/2,-w/2,-w/2, w/2, w/2,-w/2,-w/2, w/2]

    R = np.array([[np.cos(ry),0,np.sin(ry)],
                  [0,1,0],
                  [-np.sin(ry),0,np.cos(ry)]])
    corners = np.dot(np.vstack([x_c,y_c,z_c]).T, R.T)
    corners += np.array([x,y,z])
    return corners

def cam_to_velo(c, R0, Tr):
    T = R0 @ Tr
    T_inv = np.linalg.inv(T)
    c_h = np.hstack([c, np.ones((8,1))])
    return (T_inv @ c_h.T).T[:, :3]

def project_to_image(c, P2):
    c_h = np.hstack([c, np.ones((8,1))])
    p = (P2 @ c_h.T).T
    p[:,0] /= p[:,2]
    p[:,1] /= p[:,2]
    return p[:, :2].astype(int)

def draw_3d_box_on_image(img, c):
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i,j in edges:
        cv2.line(img, tuple(c[i]), tuple(c[j]), (0,0,255), 2)

def load_lidar(p):
    return np.fromfile(p, dtype=np.float32).reshape(-1,4)[:, :3]

def load_models():
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL).to(device).eval()
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    return yolo, midas, transform, device

# =========================
# MAIN
# =========================
def main():
    imgs = sorted(glob.glob(os.path.join(IMG_DIR,"*.png")))[:MAX_FRAMES]
    ids = [os.path.splitext(os.path.basename(i))[0] for i in imgs]

    yolo, midas, tfm, device = load_models()

    vis = o3d.visualization.Visualizer()
    vis.create_window("LiDAR + 3D Boxes", 1280, 720)

    pcd = o3d.geometry.PointCloud()
    added = False
    paused = False
    idx = 0

    print("SPACE=pause | N=next | P=prev | Q=quit")

    while True:
        fid = ids[idx]
        img = cv2.imread(os.path.join(IMG_DIR,f"{fid}.png"))
        pts = load_lidar(os.path.join(LIDAR_DIR,f"{fid}.bin"))
        P2,R0,Tr = read_calib(os.path.join(CALIB_DIR,f"{fid}.txt"))
        objs = parse_kitti_labels(os.path.join(LABEL_DIR,f"{fid}.txt"))

        # YOLO
        res = yolo(img, verbose=False)[0]

        # Depth
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = tfm(rgb).to(device)
        with torch.no_grad():
            d = midas(inp).squeeze().cpu().numpy()
        d = (d-d.min())/(d.max()-d.min()+1e-6)

        for b in res.boxes:
            x1,y1,x2,y2 = b.xyxy[0].int().tolist()
            dep = np.median(d[y1:y2,x1:x2])
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,f"depth~{dep:.2f}",(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

        for o in objs:
            c_cam = corners_3d_in_camera(o)
            c2d = project_to_image(c_cam,P2)
            draw_3d_box_on_image(img,c2d)

        pcd.points = o3d.utility.Vector3dVector(pts)
        if not added:
            vis.add_geometry(pcd)
            added=True
        else:
            vis.update_geometry(pcd)

        cv2.imshow("Camera + 3D Boxes", img)
        vis.poll_events()
        vis.update_renderer()

        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key in [27,ord('q')]: break
        elif key==ord(' '): paused=not paused
        elif key==ord('n'): idx=min(idx+1,len(ids)-1)
        elif key==ord('p'): idx=max(idx-1,0)
        elif not paused:
            time.sleep(1.0/PLAYBACK_FPS)   # ⏱️ DELAY ADDED
            idx=min(idx+1,len(ids)-1)

    cv2.destroyAllWindows()
    vis.destroy_window()

if __name__=="__main__":
    main()
