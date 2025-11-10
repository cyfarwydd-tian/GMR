#!/usr/bin/env python3
import argparse, sys, numpy as np, torch

# ---------- math helpers ----------
def mat6d_to_rotmat(x):
    # x: (..., 6) -> rotmat (..., 3, 3)
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)

def rotmat_to_aa(R):
    # R: (..., 3, 3) -> axis-angle (..., 3)
    # uses pytorch's rotation API if available
    try:
        from torch.nn.functional import normalize
        # trace formula -> angle + axis
        # robust conversion
        # Prefer torchvision's so3_log_map if available
        import math
        eps = 1e-6
        B = R.shape[:-2]
        Rf = R.reshape(-1, 3, 3)
        aa_list = []
        for r in Rf:
            cos_theta = ((r.trace() - 1) / 2).clamp(-1 + eps, 1 - eps)
            theta = torch.arccos(cos_theta)
            if theta < 1e-6:
                aa = torch.zeros(3, dtype=R.dtype)
            else:
                v = torch.tensor([r[2,1] - r[1,2], r[0,2] - r[2,0], r[1,0] - r[0,1]], dtype=R.dtype) / (2*torch.sin(theta))
                aa = v * theta
            aa_list.append(aa)
        aa = torch.stack(aa_list, dim=0).reshape(*B, 3)
        return aa
    except Exception:
        # fallback: use scipy if present
        try:
            from scipy.spatial.transform import Rotation as Rsc
            aa = Rsc.from_matrix(R.detach().cpu().numpy()).as_rotvec()
            return torch.from_numpy(aa).to(R)
        except Exception as e:
            raise RuntimeError("Cannot convert rotmat to axis-angle; install scipy or use torch>=2.0") from e

def maybe_to_aa(x):
    """
    Accepts one of:
      - axis-angle (..., 3)
      - 6D (..., 6)  (Zhou et al. repr)
      - rotmat (..., 3, 3)
    Returns axis-angle (..., 3)
    """
    if x is None:
        return None
    if x.shape[-1] == 3 and x.ndim >= 1:
        return x  # already aa
    if x.shape[-1] == 6:
        R = mat6d_to_rotmat(x)
        return rotmat_to_aa(R)
    if x.shape[-2:] == (3, 3):
        return rotmat_to_aa(x)
    raise ValueError(f"Unknown rotation format with shape {tuple(x.shape)}")

# ---------- key normalization ----------
ALIASES = {
    # global / body
    "global_orient": ["global_orient", "root_orient", "orient", "pose_root", "pose[:3]"],
    "body_pose":     ["body_pose", "pose_body", "body", "body_pose_aa", "pose[3:66]"],
    "transl":        ["transl", "translation", "trans", "transl_root"],
    "betas":         ["betas", "shape", "shape_params"],
    "expression":    ["expression", "exp", "expr", "expr_params"],
    # jaw & eyes
    "jaw_pose":      ["jaw_pose", "pose_jaw"],
    "leye_pose":     ["leye_pose", "left_eye_pose"],
    "reye_pose":     ["reye_pose", "right_eye_pose"],
    # hands (axis-angle of 15 joints => 45 dims)
    "left_hand_pose":  ["left_hand_pose", "lh_pose", "handl", "hand_pose_left"],
    "right_hand_pose": ["right_hand_pose", "rh_pose", "handr", "hand_pose_right"],
    # sometimes hands are concatenated
    "hand_pose":       ["hand_pose", "hands", "both_hand_pose"],
    # misc
    "gender":        ["gender", "gndr"],
    "scale":         ["scale"],
    "model_type":    ["model_type"],
}

def find_key(dct, names):
    for n in names:
        if n in dct:
            return n
    return None

def to_numpy(x):
    if x is None:
        return None
    if isinstance(x, (int, float, str)):
        return x
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    # lists etc.
    try:
        return np.array(x)
    except Exception:
        return x

def split_concat_hands(hand_pose):
    """
    If a single tensor contains both hands in axis-angle or 6D or rotmats,
    split into L/R (each (..., 45) in aa).
    Accepted shapes:
      - (..., 90) in aa
      - (..., 30, 3) in aa
      - (..., 2, 15, 3, 3) rotmats
      - (..., 2, 15, 6) 6D
    """
    shp = hand_pose.shape
    if shp[-1] == 90:
        lh = hand_pose[..., :45]
        rh = hand_pose[..., 45:]
        return lh, rh
    if shp[-2:] == (30, 3):  # (.., 30, 3) -> concatenate of 2x15 in aa
        lh = hand_pose[..., :15, :].reshape(*shp[:-2], 45)
        rh = hand_pose[..., 15:, :].reshape(*shp[:-2], 45)
        return lh, rh
    if shp[-2:] == (15, 3, 3):
        # could be (..., 2, 15, 3, 3) or (..., 30, 3, 3)
        if len(shp) >= 4 and shp[-4] == 2:
            L = hand_pose[..., 0, :, :, :]
            R = hand_pose[..., 1, :, :, :]
        else:
            L = hand_pose[..., :15, :, :]
            R = hand_pose[..., 15:, :, :]
        lh = maybe_to_aa(L).reshape(*L.shape[:-2], 45)
        rh = maybe_to_aa(R).reshape(*R.shape[:-2], 45)
        return lh, rh
    if shp[-1] == 6 and (len(shp) >= 3 and (shp[-2] == 30 or (shp[-3] == 2 and shp[-2] == 15))):
        if shp[-3] == 2 and shp[-2] == 15:
            L6 = hand_pose[..., 0, :, :]
            R6 = hand_pose[..., 1, :, :]
        else:
            L6 = hand_pose[..., :15, :]
            R6 = hand_pose[..., 15:, :]
        Laa = maybe_to_aa(L6).reshape(*L6.shape[:-2], 45)
        Raa = maybe_to_aa(R6).reshape(*R6.shape[:-2], 45)
        return Laa, Raa
    raise ValueError(f"Cannot split concatenated hand_pose with shape {tuple(shp)}")

def ensure_axis_angle(x, name, expected_last=3):
    x = torch.as_tensor(x) if not torch.is_tensor(x) else x
    if name in ["left_hand_pose", "right_hand_pose"]:
        # expect (..., 45) aa
        if x.shape[-1] == 45:
            return x
        # maybe (..., 15, 3)
        if x.shape[-2:] == (15, 3):
            return x.reshape(*x.shape[:-2], 45)
        # 6D or rotmat per joint
        if x.shape[-1] == 6:
            aa = maybe_to_aa(x)  # (..., 15, 3)
            return aa.reshape(*aa.shape[:-2], 45)
        if x.shape[-2:] == (3, 3):
            aa = maybe_to_aa(x)  # (..., 15, 3)
            return aa.reshape(*aa.shape[:-2], 45)
        raise ValueError(f"{name} has unsupported shape {tuple(x.shape)}")
    else:
        # expect (..., 3)
        return maybe_to_aa(x)

def broadcast_time_dim(*xs):
    """Make all tensors share the same leading time dimension T if any has it."""
    shapes = [tuple(x.shape) if x is not None else None for x in xs]
    Ts = [s[0] for s in shapes if s is not None and len(s) >= 2 and s[0] > 1]  # consider (T, *)
    T = max(Ts) if Ts else None
    out = []
    for x in xs:
        if x is None:
            out.append(None)
            continue
        if T is None:
            out.append(x)
            continue
        # if x is single-frame (...), expand to (T, ...)
        if len(x.shape) == 1 or (len(x.shape) >= 2 and x.shape[0] != T):
            x = x.unsqueeze(0) if len(x.shape) == 1 else x
            if x.shape[0] == 1:
                x = x.expand(T, *x.shape[1:])
        out.append(x)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help=".pt file path")
    ap.add_argument("--output", "-o", required=True, help=".npz output path")
    ap.add_argument("--default_gender", choices=["neutral","male","female"], default="neutral")
    args = ap.parse_args()

    data = torch.load(args.input, map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data and isinstance(data["state_dict"], dict):
        data = data["state_dict"]

    # sometimes the params live under nested keys
    # try a few common containers
    for k in ["params", "smplx", "smplx_params", "body", "result", "output", "pred"]:
        if isinstance(data, dict) and k in data and isinstance(data[k], dict):
            data = data[k]

    # find keys
    def get(names):
        key = find_key(data, names)
        return data.get(key) if key is not None else None

    global_orient = get(ALIASES["global_orient"])
    body_pose     = get(ALIASES["body_pose"])
    transl        = get(ALIASES["transl"])
    betas         = get(ALIASES["betas"])
    expression    = get(ALIASES["expression"])
    jaw_pose      = get(ALIASES["jaw_pose"])
    leye_pose     = get(ALIASES["leye_pose"])
    reye_pose     = get(ALIASES["reye_pose"])
    lhand         = get(ALIASES["left_hand_pose"])
    rhand         = get(ALIASES["right_hand_pose"])
    hand_concat   = get(ALIASES["hand_pose"])
    gender        = get(ALIASES["gender"])
    scale         = get(ALIASES["scale"])
    model_type    = get(ALIASES["model_type"]) or "smplx"

    # handle concatenated hands
    if (lhand is None or rhand is None) and hand_concat is not None:
        try:
            lhand, rhand = split_concat_hands(torch.as_tensor(hand_concat))
        except Exception as e:
            print(f"[warn] Could not split hand_pose automatically: {e}")

    # convert all rotations to axis-angle
    if global_orient is not None:
        global_orient = ensure_axis_angle(torch.as_tensor(global_orient), "global_orient")
    if body_pose is not None:
        body_pose = torch.as_tensor(body_pose)
        # body can be (..., 63) aa or (..., 21, 3) aa, or 6D/rotmat per joint
        if body_pose.shape[-1] == 63:
            pass  # already aa
        elif body_pose.shape[-2:] == (21, 3):
            body_pose = body_pose.reshape(*body_pose.shape[:-2], 63)
        elif body_pose.shape[-1] == 6:
            # assume (..., 21, 6)
            aa = maybe_to_aa(body_pose)
            body_pose = aa.reshape(*aa.shape[:-2], 63)
        elif body_pose.shape[-2:] == (3, 3):
            # assume (..., 21, 3, 3)
            aa = maybe_to_aa(body_pose)
            body_pose = aa.reshape(*aa.shape[:-2], 63)
        else:
            raise ValueError(f"body_pose has unsupported shape {tuple(body_pose.shape)}")

    if jaw_pose is not None:
        jaw_pose = ensure_axis_angle(torch.as_tensor(jaw_pose), "jaw_pose")
    if leye_pose is not None:
        leye_pose = ensure_axis_angle(torch.as_tensor(leye_pose), "leye_pose")
    if reye_pose is not None:
        reye_pose = ensure_axis_angle(torch.as_tensor(reye_pose), "reye_pose")
    if lhand is not None:
        lhand = ensure_axis_angle(torch.as_tensor(lhand), "left_hand_pose")
    if rhand is not None:
        rhand = ensure_axis_angle(torch.as_tensor(rhand), "right_hand_pose")

    if transl is not None:
        transl = torch.as_tensor(transl).reshape(-1, 3) if torch.as_tensor(transl).numel()%3==0 else torch.as_tensor(transl)
    if betas is not None:
        betas = torch.as_tensor(betas)
    if expression is not None:
        expression = torch.as_tensor(expression)
    if scale is not None:
        scale = torch.as_tensor(scale)

    # broadcast time dimension if needed
    global_orient, body_pose, jaw_pose, leye_pose, reye_pose, lhand, rhand, transl = \
        broadcast_time_dim(global_orient, body_pose, jaw_pose, leye_pose, reye_pose, lhand, rhand, transl)

    # gender default
    if gender is None:
        gender = args.default_gender
    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    if isinstance(gender, torch.Tensor) and gender.ndim == 0:
        gender = str(gender.item())

    # final dict in "standard" smplx npz schema (axis-angle)
    out = {}
    if betas is not None:      out["betas"] = to_numpy(betas)
    if expression is not None: out["expression"] = to_numpy(expression)
    if global_orient is not None: out["global_orient"] = to_numpy(global_orient)
    if body_pose is not None:     out["body_pose"] = to_numpy(body_pose)
    if jaw_pose is not None:      out["jaw_pose"] = to_numpy(jaw_pose)
    if leye_pose is not None:     out["leye_pose"] = to_numpy(leye_pose)
    if reye_pose is not None:     out["reye_pose"] = to_numpy(reye_pose)
    if lhand is not None:         out["left_hand_pose"] = to_numpy(lhand)
    if rhand is not None:         out["right_hand_pose"] = to_numpy(rhand)
    if transl is not None:        out["transl"] = to_numpy(transl)
    if scale is not None:         out["scale"] = to_numpy(scale)
    out["gender"] = np.array(gender)  # keep as string
    out["model_type"] = np.array(model_type)

    # sanity checks
    def lastdim(x): return None if x is None else x.shape[-1]
    if "global_orient" in out and out["global_orient"].shape[-1] != 3: raise ValueError("global_orient must be axis-angle (…,3)")
    if "body_pose" in out and out["body_pose"].shape[-1] != 63: raise ValueError("body_pose must be axis-angle of 21 joints (…,63)")
    if "left_hand_pose" in out and out["left_hand_pose"].shape[-1] != 45: raise ValueError("left_hand_pose must be (…,45)")
    if "right_hand_pose" in out and out["right_hand_pose"].shape[-1] != 45: raise ValueError("right_hand_pose must be (…,45)")
    if "jaw_pose" in out and out["jaw_pose"].shape[-1] != 3: raise ValueError("jaw_pose must be (…,3)")
    if "leye_pose" in out and out["leye_pose"].shape[-1] != 3: raise ValueError("leye_pose must be (…,3)")
    if "reye_pose" in out and out["reye_pose"].shape[-1] != 3: raise ValueError("reye_pose must be (…,3)")
    if "transl" in out and out["transl"].shape[-1] != 3: raise ValueError("transl must be (…,3)")

    # save compressed npz
    np.savez_compressed(args.output, **out)
    print(f"[ok] saved SMPL-X npz -> {args.output}")
    # small preview
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  {k}: {type(v)}")

if __name__ == "__main__":
    main()



