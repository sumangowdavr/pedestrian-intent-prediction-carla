import os
import cv2
import numpy as np
import pandas as pd
import argparse
from glob import glob
from tqdm import tqdm

# ----------------------------- UTILS -----------------------------

def extract_and_match(img1, img2, detector, matcher, ratio=0.75):
    """
    Detects keypoints+descriptors in img1/img2, matches with ratio test.
    Returns matched pt arrays pts1, pts2, plus raw kp1, kp2.
    """
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return np.empty((0,2)), np.empty((0,2)), kp1, kp2

    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m0 for m0,m1 in raw if m0.distance < ratio*m1.distance]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2, kp1, kp2

def recover_pose(pts1, pts2, K):
    """
    Given matched pts1/pts2 and intrinsics K,
    compute Essential matrix → recover R, t.
    """
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None:
        return None, None, None
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, mask_pose

def save_match_vis(img1, img2, kp1, kp2, matches, out_path, max_draw=50):
    """
    Draws up to max_draw matches between img1/img2 and writes to out_path.
    """
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches[:max_draw], None,
        matchColor=(0,255,0), singlePointColor=(255,0,0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(out_path, vis)

def overlay_pose_arrow(rgb, origin, t, scale=100.0, color=(0,0,255), thickness=2):
    """
    Draws an arrow on rgb at pixel 'origin' pointing in the direction of the motion vector t (in world coords).
    We map t.x→right, t.z→up on image for a rough illustration.
    """
    x0,y0 = origin
    dx = int(scale * ( t[0,0]/(abs(t[2,0])+1e-6) ))  # X vs Z
    dy = int(scale * (-(t[2,0]/(abs(t[2,0])+1e-6))))  # invert for image coords
    cv2.arrowedLine(rgb, (x0,y0), (x0+dx, y0+dy), color, thickness, tipLength=0.2)
    return rgb

# ----------------------------- MAIN -----------------------------

def main(args):
    # 1) Gather all RGB frames
    img_paths = sorted(glob(os.path.join(args.rgb_dir, '*.png')))
    assert len(img_paths) >= 2, "Need at least two frames in --rgb_dir"

    # 2) Camera intrinsics (tweak for your Carla rig)
    fx = fy = 700.0
    cx = cy = 640.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)

    # 3) ORB & BFMatcher setup
    orb = cv2.ORB_create(5000)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 4) Prepare output folders
    vis_match_dir = "feature_matches"
    vis_pose_dir  = "pose_vis"
    for d in (vis_match_dir, vis_pose_dir, os.path.dirname(args.output_csv)):
        os.makedirs(d, exist_ok=True)

    # 5) Initialize pose
    trajectory = []
    cur_R = np.eye(3)
    cur_t = np.zeros((3,1))
    trajectory.append([0.0,0.0,0.0])  # frame 0

    # 6) Iterate frames
    prev_gray = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    for idx in tqdm(range(1, len(img_paths)), desc="VO frames"):
        curr_gray = cv2.imread(img_paths[idx], cv2.IMREAD_GRAYSCALE)
        rgb_color  = cv2.imread(img_paths[idx], cv2.IMREAD_COLOR)

        # a) Extract & match
        pts1, pts2, kp1, kp2 = extract_and_match(prev_gray, curr_gray, orb, bf,
                                                 ratio=0.75)

        # b) Save match viz
        matches = bf.knnMatch(orb.detectAndCompute(prev_gray,None)[1],
                              orb.detectAndCompute(curr_gray,None)[1], k=2)
        m_vis_path = os.path.join(vis_match_dir, f"{idx-1:06d}_{idx:06d}.png")
        save_match_vis(prev_gray, curr_gray, kp1, kp2, matches, m_vis_path)

        # c) If not enough correspondences → skip
        if len(pts1) < 8:
            trajectory.append([cur_t[0,0], cur_t[1,0], cur_t[2,0]])
            prev_gray = curr_gray
            continue

        # d) Recover pose
        R, t, mask = recover_pose(pts1, pts2, K)
        if R is None:
            trajectory.append([cur_t[0,0], cur_t[1,0], cur_t[2,0]])
            prev_gray = curr_gray
            continue

        # e) Chain new pose: T_new = T_old * inv([R|t])
        cur_t = cur_t + cur_R.dot(t)
        cur_R = R.dot(cur_R)

        trajectory.append([cur_t[0,0], cur_t[1,0], cur_t[2,0]])

        # f) Visualize pose arrow at image center
        h,w = rgb_color.shape[:2]
        origin = (w//2, h//2)
        vis_rgb = overlay_pose_arrow(rgb_color.copy(), origin, t,
                                     scale=args.arrow_scale,
                                     color=(0,0,255), thickness=2)
        cv2.imwrite(os.path.join(vis_pose_dir, f"{idx:06d}.png"), vis_rgb)

        prev_gray = curr_gray

    # 7) Save trajectory CSV
    df = pd.DataFrame(trajectory, columns=['x','y','z'])
    df.index.name = 'frame'
    df.to_csv(args.output_csv)
    print(f"✅ Trajectory saved to {args.output_csv}")

    # 8) Plot top-down X vs Z
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,6))
        plt.plot(df['x'], df['z'], '-o', markersize=3, label='VO path')
        plt.title("Top-down Trajectory (X vs Z)")
        plt.xlabel("X (m)")
        plt.ylabel("Z (m)")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

# --------------------------- ENTRYPOINT --------------------------

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Monocular Visual Odometry + Visualization"
    )
    parser.add_argument(
        "--rgb_dir",    type=str,
        help="Folder containing input RGB .png frames", required=True
    )
    parser.add_argument(
        "--output_csv", type=str,
        help="Output trajectory CSV (frame,x,y,z)", default="vo_trajectory.csv"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Show top-down trajectory plot at end"
    )
    parser.add_argument(
        "--arrow_scale", type=float, default=200.0,
        help="Scale factor for overlay arrows"
    )
    args = parser.parse_args()
    main(args)
