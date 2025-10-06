#!/usr/bin/env python3
"""
bvh_skeleton_probe.py
---------------------
Auto-detect a BVH's HIERARCHY, print a clean joint tree, and suggest a Bandai->GMR name map.
It also proposes Left/Right toe fallbacks.

Usage:
  python bvh_skeleton_probe.py /path/to/file.bvh --write-map bandai_name_map.py

Outputs:
  - Pretty-printed hierarchy to stdout
  - Suggested mapping (print + optional write to Python module)
  - Stats: fps, rough unit guess (via hip height), joint count

Heuristics:
  - Left/Right detection via prefixes: ["Left", "Right", "L", "R"]
  - Toe candidates: name contains any of ["Toe", "ToeBase", "toes", "toe"]
  - Canonical GMR names list is included below (edit as needed).

This script does not require SciPy; it parses BVH header (HIERARCHY block) directly.
"""

import re
import argparse
from collections import defaultdict

CANONICAL_GMR_SET = [
    # Core torso
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head",
    # Left arm
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    # Right arm
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    # Left leg
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    # Right leg
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
]

LEFT_PREFIXES  = ["Left", "L"]
RIGHT_PREFIXES = ["Right", "R"]
TOE_HINTS = ["ToeBase", "Toe", "toe", "toes"]

def read_bvh_header(path):
    lines = open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
    in_hier = False
    in_motion = False
    header_lines = []
    motion_lines = []
    for ln in lines:
        if ln.strip().startswith("HIERARCHY"):
            in_hier = True
            continue
        if ln.strip().startswith("MOTION"):
            in_motion = True
            in_hier = False
        if in_motion:
            motion_lines.append(ln)
        elif in_hier:
            header_lines.append(ln)

    # parse fps from motion block
    fps = None
    for ln in motion_lines:
        m = re.search(r"Frame Time:\s*([0-9\.eE+-]+)", ln)
        if m:
            dt = float(m.group(1))
            if dt > 0:
                fps = round(1.0/dt)
            break
    return header_lines, fps

def parse_hierarchy(header_lines):
    """
    Build a simple tree of joints. Returns (root_name, children_dict, offsets_dict, order_list)
    order_list is the appearance order of JOINT/ROOT for reproducibility.
    """
    children = defaultdict(list)
    offsets = {}
    order = []
    stack = []
    current = None

    # Regex helpers
    re_joint = re.compile(r"^\s*(ROOT|JOINT)\s+([A-Za-z0-9_\-\.]+)")
    re_offset = re.compile(r"^\s*OFFSET\s+([0-9\.\-eE]+)\s+([0-9\.\-eE]+)\s+([0-9\.\-eE]+)")
    re_end = re.compile(r"^\s*End Site")

    root_name = None

    for ln in header_lines:
        m = re_joint.match(ln)
        if m:
            typ, name = m.group(1), m.group(2)
            order.append(name)
            if current is not None:
                children[current].append(name)
            else:
                root_name = name
            stack.append(name)
            current = name
            continue
        if re_end.match(ln):
            # consume a matching brace later
            continue
        if "{" in ln:
            continue
        if "}" in ln:
            if stack:
                stack.pop()
                current = stack[-1] if stack else None
            continue
        mo = re_offset.match(ln)
        if mo and current:
            offsets[current] = tuple(float(mo.group(i)) for i in range(1,4))

    return root_name, dict(children), offsets, order

def print_tree(root, children, level=0):
    indent = "  " * level
    print(f"{indent}- {root}")
    for ch in children.get(root, []):
        print_tree(ch, children, level+1)

def is_left(name):
    return any(name.startswith(p) for p in LEFT_PREFIXES)

def is_right(name):
    return any(name.startswith(p) for p in RIGHT_PREFIXES)

def suggest_map(order):
    """
    Very lightweight heuristic mapping based on typical humanoid conventions.
    You should review and edit the produced map.
    """
    name_map = {}
    # Pass-through for identical names
    for n in order:
        if n in CANONICAL_GMR_SET:
            name_map[n] = n

    # Common variants
    variants = [
        ("LeftToeBase", "LeftToe"),
        ("RightToeBase", "RightToe"),
        ("LeftUpLegRoll", "LeftUpLeg"),
        ("RightUpLegRoll", "RightUpLeg"),
        ("LeftArmRoll", "LeftArm"),
        ("RightArmRoll", "RightArm"),
        ("LeftForeArmRoll", "LeftForeArm"),
        ("RightForeArmRoll", "RightForeArm"),
    ]
    for src, dst in variants:
        if src in order and dst in CANONICAL_GMR_SET:
            name_map[src] = dst

    # Try to catch shoulder variants
    for n in order:
        low = n.lower()
        if "clavicle" in low and "LeftShoulder" not in name_map and is_left(n):
            name_map[n] = "LeftShoulder"
        if "clavicle" in low and "RightShoulder" not in name_map and is_right(n):
            name_map[n] = "RightShoulder"

    # Heuristic toe pick
    toes_left  = [n for n in order if is_left(n) and any(h in n for h in TOE_HINTS)]
    toes_right = [n for n in order if is_right(n) and any(h in n for h in TOE_HINTS)]
    if toes_left and "LeftToe" not in name_map:
        name_map[toes_left[-1]] = "LeftToe"
    if toes_right and "RightToe" not in name_map:
        name_map[toes_right[-1]] = "RightToe"

    return name_map

def rough_unit_guess(offsets):
    """Return a crude guess of unit by looking at pelvis->head vertical span."""
    # try to find Hips and Head
    hips = offsets.get("Hips", None)
    # This is too naive because offsets are local, but we just want a magnitude scale.
    # We'll take any offset magnitudes as hints.
    mags = [abs(v) for o in offsets.values() for v in o]
    if not mags:
        return "unknown"
    avg = sum(mags)/len(mags)
    # Typically in BVH (cm), offsets are on tens; in meters they'd be <2.
    return "cm-like" if avg > 2.0 else "m-like"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bvh", help="Path to a BVH file")
    ap.add_argument("--write-map", help="Write suggested map to a Python module (e.g., bandai_name_map.py)")
    args = ap.parse_args()

    header_lines, fps = read_bvh_header(args.bvh)
    root, children, offsets, order = parse_hierarchy(header_lines)

    print("== BVH Hierarchy (JOINT order) ==")
    print_tree(root, children)
    print("\n== Stats ==")
    print(f"Joints: {len(order)}")
    print(f"FPS   : {fps if fps else 'unknown'}")
    print(f"Unit? : {rough_unit_guess(offsets)} (heuristic)")

    m = suggest_map(order)
    print("\n== Suggested Bandai->GMR name map ==")
    for k, v in m.items():
        print(f"{k}  ->  {v}")

    # Toe fallbacks
    ltoe = next((k for k,v in m.items() if v == "LeftToe"), None)
    rtoe = next((k for k,v in m.items() if v == "RightToe"), None)
    print("\n== Toe Fallbacks ==")
    print(f"LeftToe  source: {ltoe or 'NOT FOUND (fallback to LeftFoot orientation)'}")
    print(f"RightToe source: {rtoe or 'NOT FOUND (fallback to RightFoot orientation)'}")

    if args.write_map:
        with open(args.write_map, "w", encoding="utf-8") as f:
            f.write("# Auto-generated by bvh_skeleton_probe.py\n")
            f.write("NAME_MAP_BANDAI2GMR = {\n")
            for k, v in m.items():
                f.write(f"    {repr(k)}: {repr(v)},\n")
            f.write("}\n")
        print(f"\n[Saved] Suggested map -> {args.write_map}")

if __name__ == "__main__":
    main()
