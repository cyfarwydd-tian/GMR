import numpy as np, shutil, os

p = "./assets/output/Reference_human_2.npz"
bak = p + ".bak"
if not os.path.exists(bak):
    shutil.copy2(p, bak)

d = np.load(p)
out = {k: d[k] for k in d.files}

def alias(src_key, dst_key):
    if src_key in d.files and dst_key not in out:
        out[dst_key] = d[src_key]

# 标准 -> GMR 键名
alias("body_pose", "pose_body")          # (T,63)
alias("global_orient", "root_orient")    # (T,3)
alias("transl", "trans")                  # (T,3)
alias("left_hand_pose", "handl")          # (T,45)
alias("right_hand_pose", "handr")         # (T,45)
alias("expression", "expr")               # (T,D)

# 单帧补成 (1, …)
for k in ["pose_body","root_orient","trans","handl","handr","jaw_pose","leye_pose","reye_pose"]:
    if k in out:
        arr = out[k]
        if arr.ndim == 1:
            arr = arr[None, ...]
        out[k] = arr.astype(np.float32) if arr.dtype.kind in "fc" else arr

np.savez_compressed(p, **out)
print("✔ fixed and saved:", p)
print("keys:", sorted(out.keys()))




