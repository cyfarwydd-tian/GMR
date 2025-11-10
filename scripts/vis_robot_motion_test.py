from general_motion_retargeting import RobotMotionViewer, load_robot_motion
import argparse, os, math, numpy as np

def polar_to_cartesian(azim_deg, elev_deg, dist, target):
    az = math.radians(azim_deg); el = math.radians(elev_deg)
    x = dist * math.cos(el) * math.cos(az) + target[0]
    y = dist * math.cos(el) * math.sin(az) + target[1]
    z = dist * math.sin(el) + target[2]
    return np.array([x, y, z], dtype=float)

def try_set_camera(viewer, eye, target):
    # 兼容不同Viewer实现
    if hasattr(viewer, "set_camera") and callable(viewer.set_camera):
        viewer.set_camera(eye=eye, target=target); return True
    if hasattr(viewer, "set_camera_pose") and callable(viewer.set_camera_pose):
        viewer.set_camera_pose(eye=eye, target=target); return True
    if hasattr(viewer, "set_camera_lookat") and callable(viewer.set_camera_lookat):
        viewer.set_camera_lookat(target=target, eye=eye); return True
    # 最后兜底：直接改属性
    if hasattr(viewer, "camera_eye") and hasattr(viewer, "camera_target"):
        try:
            viewer.camera_eye = np.array(eye); viewer.camera_target = np.array(target); return True
        except Exception:
            pass
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default="unitree_g1")
    parser.add_argument("--robot_motion_path", type=str, required=True)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    # <<< 相机参数（正前方=0°，背后=180°，左=90°，右=-90°）
    parser.add_argument("--cam-azim", type=float, default=0.0)
    parser.add_argument("--cam-elev", type=float, default=15.0)
    parser.add_argument("--cam-dist", type=float, default=4.0)
    parser.add_argument("--cam-target", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    args = parser.parse_args()

    if not os.path.exists(args.robot_motion_path):
        raise FileNotFoundError(f"Motion file {args.robot_motion_path} not found")

    (motion_data, motion_fps, motion_root_pos, motion_root_rot,
     motion_dof_pos, motion_local_body_pos, motion_link_body_list) = load_robot_motion(args.robot_motion_path)

    os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)

    env = RobotMotionViewer(robot_type=args.robot,
                            motion_fps=motion_fps,
                            camera_follow=False,                 # 先关掉
                            record_video=args.record_video,
                            video_path=args.video_path)

    # <<< 初始化相机
    cam_target = np.array(args.cam_target, dtype=float)
    eye = polar_to_cartesian(args.cam_azim, args.cam_elev, max(0.1, args.cam_dist), cam_target)
    try_set_camera(env, eye, cam_target)

    # <<< 尽量彻底地关闭可能的跟随开关/模式（如果有的话）
    for name in ("camera_follow", "follow", "follow_target", "enable_follow"):
        if hasattr(env, name):
            try:
                setattr(env, name, False)
            except Exception:
                pass
    for name in ("set_camera_follow", "enable_camera_follow", "set_follow"):
        if hasattr(env, name) and callable(getattr(env, name)):
            try:
                getattr(env, name)(False)
            except Exception:
                pass

    frame_idx, T = 0, len(motion_root_pos)
    try:
        while True:
            env.step(motion_root_pos[frame_idx],
                     motion_root_rot[frame_idx],
                     motion_dof_pos[frame_idx],
                     rate_limit=True)

            # <<< 关键：在 step() 之后强制把相机拉回你要的视角
            try_set_camera(env, eye, cam_target)

            frame_idx += 1
            if frame_idx >= T:
                frame_idx = 0
    finally:
        if hasattr(env, "close"):
            env.close()
