# Copyright (c) 2025
# SPDX-License-Identifier: MIT

import os
import sys
import numpy as np
import torch
import cv2

from .process import (
    comfy_image_to_numpy,
    comfy_mask_to_numpy,
    numpy_to_comfy_image,
    _load_sam3d_model,
)

_KIJAI_FOUND = False
_KIJAI_ERROR = None
AAPoseMeta = None
draw_aapose_by_meta_new = None
draw_face_kp = None
FACE_CUSTOM_STYLE = None
padding_resize = None

def _locate_kijai_root():
    _here = os.path.dirname(os.path.abspath(__file__))
    _custom_nodes_dir = os.path.abspath(os.path.join(_here, "..", "..", ".."))
    for cand in ("ComfyUI-WanAnimatePreprocess", "ComfyUI-WanAnimatePreprocessV2"):
        p = os.path.join(_custom_nodes_dir, cand)
        if os.path.isdir(p):
            return p
    return None

try:
    import importlib.util

    _kijai_root = _locate_kijai_root()
    if _kijai_root is None:
        raise ImportError(
            "ComfyUI-WanAnimatePreprocess (or V2) directory not found next to "
            "ComfyUI-SAM3DBody in custom_nodes/."
        )


    def _load_module_from_path(unique_name, file_path):
        spec = importlib.util.spec_from_file_location(unique_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot build spec for {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        spec.loader.exec_module(mod)
        return mod

    _pose2d_path = os.path.join(_kijai_root, "pose_utils", "pose2d_utils.py")
    _humanvis_path = os.path.join(_kijai_root, "pose_utils", "human_visualization.py")
    _utils_path = os.path.join(_kijai_root, "utils.py")

    for p in (_pose2d_path, _humanvis_path, _utils_path):
        if not os.path.isfile(p):
            raise ImportError(f"kijai file missing: {p}")

    _NS = "_sam3dbody_kijai"

    _pkg_name = f"{_NS}.pose_utils"
    _pkg_spec = importlib.util.spec_from_loader(_pkg_name, loader=None, is_package=True)
    _pkg_mod = importlib.util.module_from_spec(_pkg_spec)
    _pkg_mod.__path__ = [os.path.join(_kijai_root, "pose_utils")]
    sys.modules[_pkg_name] = _pkg_mod

    _root_pkg = _NS
    _root_spec = importlib.util.spec_from_loader(_root_pkg, loader=None, is_package=True)
    _root_mod = importlib.util.module_from_spec(_root_spec)
    _root_mod.__path__ = [_kijai_root]
    sys.modules[_root_pkg] = _root_mod

    _pose2d_mod = _load_module_from_path(f"{_NS}.pose_utils.pose2d_utils", _pose2d_path)
    setattr(_pkg_mod, "pose2d_utils", _pose2d_mod)

    _humanvis_mod = _load_module_from_path(f"{_NS}.pose_utils.human_visualization", _humanvis_path)
    setattr(_pkg_mod, "human_visualization", _humanvis_mod)

    _utils_mod = _load_module_from_path(f"{_NS}.utils", _utils_path)
    setattr(_root_mod, "utils", _utils_mod)

    AAPoseMeta = _pose2d_mod.AAPoseMeta
    draw_aapose_by_meta_new = _humanvis_mod.draw_aapose_by_meta_new
    draw_face_kp = _humanvis_mod.draw_face_kp
    FACE_CUSTOM_STYLE = _humanvis_mod.FACE_CUSTOM_STYLE
    padding_resize = _utils_mod.padding_resize


    _KIJAI_FOUND = True
except Exception as e:
    import traceback as _tb
    _KIJAI_ERROR = f"{type(e).__name__}: {e}\n{_tb.format_exc()}"

# MHR70 -> Wan20 body mapping
MHR_TO_WAN20 = np.array([
    0,   # Wan 0  Nose
    69,  # Wan 1  Neck
    6,   # Wan 2  RShoulder
    8,   # Wan 3  RElbow
    41,  # Wan 4  RWrist
    5,   # Wan 5  LShoulder
    7,   # Wan 6  LElbow
    62,  # Wan 7  LWrist
    10,  # Wan 8  RHip
    12,  # Wan 9  RKnee
    14,  # Wan 10 RAnkle
    9,   # Wan 11 LHip
    11,  # Wan 12 LKnee
    13,  # Wan 13 LAnkle
    2,   # Wan 14 REye
    1,   # Wan 15 LEye
    4,   # Wan 16 REar
    3,   # Wan 17 LEar
    15,  # Wan 18 LToe
    18,  # Wan 19 RToe
], dtype=np.int64)

# MHR70 -> OpenPose 21-joint hand mapping
MHR_TO_OPENPOSE_LHAND = np.array([
    62, 45, 44, 43, 42, 49, 48, 47, 46, 53, 52, 51, 50, 57, 56, 55, 54, 61, 60, 59, 58,
], dtype=np.int64)
MHR_TO_OPENPOSE_RHAND = np.array([
    41, 24, 23, 22, 21, 28, 27, 26, 25, 32, 31, 30, 29, 36, 35, 34, 33, 40, 39, 38, 37,
], dtype=np.int64)

_LHIP_IDX = 9
_RHIP_IDX = 10

# MHR_idx -> AA20_idx (kijai POSEDATA body layout)
# A joint enters the keypoint-prompt set when its POSEDATA confidence clears
# the threshold and leaves only when it drops below threshold x this. Without
# the band a joint sitting near the threshold joins and leaves between frames,
# and the refine head then solves a different problem each frame - measured as
# 20 px/frame swings while the driver stood still. Not a setting: with a steady
# confidence signal the band changes nothing.
_KP_PROMPT_RELEASE = 0.5

_AA_TO_MHR = {
    0: 0,    # nose
    1: 15,   # leye
    2: 14,   # reye
    3: 17,   # lear
    4: 16,   # rear
    5: 5,    # lshoulder
    6: 2,    # rshoulder
    7: 6,    # lelbow
    8: 3,    # relbow
    9: 11,   # lhip
    10: 8,   # rhip
    11: 12,  # lknee
    12: 9,   # rknee
    13: 13,  # lankle
    14: 10,  # rankle
    15: 18,  # lbigtoe
    18: 19,  # rbigtoe
    41: 4,   # rwrist
    62: 7,   # lwrist
}

def _perspective_project(j3d_cam, cam_int):
    z = j3d_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-8, np.where(z < 0, -1e-8, 1e-8), z)
    y = j3d_cam / z
    proj = (cam_int @ y.T).T


    return proj[:, :2]

# ---------------------------------------------------------------------------
# Appearance-transfer helpers (face geometry follow, swap-silhouette mask,
# clothing-volume measurement). Only numpy + cv2; no torch on the hot path.
# ---------------------------------------------------------------------------

# Rigid translation chains for shoulder/hip spread widening. When a shoulder
# (MHR 5/6) or hip (MHR 9/10) is pushed outward, the whole distal chain moves
# by the same delta so limb shape is preserved. Hand ranges are the MHR mesh
# hand points (L 42-62, R 21-41); they are normally overwritten by driver
# POSEDATA hands, which re-attach to the (already widened) swap wrist.
_ARM_L_CHAIN = np.concatenate([np.array([7], dtype=np.int64),
                               np.arange(42, 63, dtype=np.int64)])
_ARM_R_CHAIN = np.concatenate([np.array([8], dtype=np.int64),
                               np.arange(21, 42, dtype=np.int64)])
_LEG_L_CHAIN = np.array([11, 13, 15], dtype=np.int64)
_LEG_R_CHAIN = np.array([12, 14, 18], dtype=np.int64)


def _project_with_focal(pts3d_cam, focal, W, H):
    """Project camera-space points with the exact SAM3D output convention:
    focal in pixels, principal point at the image center. Matches the
    pred_keypoints_2d projection in sam_3d_body/model.py (verified)."""
    z = pts3d_cam[:, 2:3]
    z = np.where(np.abs(z) < 1e-8, np.where(z < 0, -1e-8, 1e-8), z)
    x = pts3d_cam[:, 0:1] / z * float(focal) + W / 2.0
    y = pts3d_cam[:, 1:2] / z * float(focal) + H / 2.0
    return np.concatenate([x, y], axis=1)


def _chain_height(j3d, mode="full"):
    """Pose-invariant skeletal height sums. mode:
      'full'  - legs (ankle->knee->hip, averaged) + pelvis->neck + neck->nose
      'torso' - pelvis->neck + neck->nose (driver legs not on screen)
      'head'  - neck->nose only (waist-up / close framing)
    Bone lengths do not change with pose; monocular scale is shared by both
    recons of the same frame, so it cancels in the swap/driver ratio."""
    j = np.asarray(j3d, dtype=np.float64)
    pel = (j[9] + j[10]) / 2.0
    torso = np.linalg.norm(j[69] - pel)
    head = np.linalg.norm(j[0] - j[69])
    if mode == "head":
        return head
    if mode == "torso":
        return torso + head
    leg_l = np.linalg.norm(j[13] - j[11]) + np.linalg.norm(j[11] - j[9])
    leg_r = np.linalg.norm(j[14] - j[12]) + np.linalg.norm(j[12] - j[10])
    return 0.5 * (leg_l + leg_r) + torso + head


def _forearm_ratio(kp2d_swap, kp2d_dri, elbow_idx, wrist_idx):
    """Projected forearm-length ratio swap/driver: sizes the driver hand to the
    swapped body (covers both proportion betas and any height scale)."""
    try:
        s = float(np.linalg.norm(
            np.asarray(kp2d_swap[wrist_idx], dtype=np.float64)
            - np.asarray(kp2d_swap[elbow_idx], dtype=np.float64)))
        d = float(np.linalg.norm(
            np.asarray(kp2d_dri[wrist_idx], dtype=np.float64)
            - np.asarray(kp2d_dri[elbow_idx], dtype=np.float64)))
        if d > 2.0 and s > 0.5:
            return float(min(max(s / d, 0.7), 1.4))
    except Exception:
        pass
    return 1.0


def _splat_silhouette(v2d, H, W, extra_dilate=2):
    """Rasterize a projected vertex cloud into a filled binary silhouette.
    Point splat -> morphological close (kernel from body span) -> hole fill."""
    pts = np.round(np.asarray(v2d, dtype=np.float64)).astype(np.int64)
    valid = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    pts = pts[valid]
    if pts.shape[0] < 50:
        return None
    canvas = np.zeros((H, W), dtype=np.uint8)
    canvas[pts[:, 1], pts[:, 0]] = 1
    bbox_w = int(pts[:, 0].max() - pts[:, 0].min()) + 1
    bbox_h = int(pts[:, 1].max() - pts[:, 1].min()) + 1
    # Close-kernel from measured point density: bridges inter-vertex gaps
    # (close does not inflate the outline - erode returns to the boundary).
    spacing = max(float(np.sqrt((bbox_w * bbox_h) / max(pts.shape[0], 1))), 1.0)
    k = int(min(max(round(3.0 * spacing), 5), 31))
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sil = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, ker)
    ff = sil.copy()
    ffmask = np.zeros((H + 2, W + 2), dtype=np.uint8)
    corner = None
    for cy, cx in ((0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)):
        if sil[cy, cx] == 0:
            corner = (cx, cy)
            break
    if corner is not None:
        cv2.floodFill(ff, ffmask, corner, 1)
        sil = ((sil > 0) | (ff == 0)).astype(np.uint8)
    if extra_dilate > 0:
        d = int(extra_dilate) * 2 + 1
        sil = cv2.dilate(sil, np.ones((d, d), dtype=np.uint8))
    return sil


def _block_snap_mask(mask01, block=32):
    """Snap a float mask to the same NxN block grid as upstream BlockifyMask
    (max-pool per block, then upsample). Idempotent on already-blocky masks."""
    H, W = mask01.shape[:2]
    ph = (-H) % block
    pw = (-W) % block
    m = np.pad(mask01, ((0, ph), (0, pw)), mode="edge")
    Hb, Wb = m.shape[0] // block, m.shape[1] // block
    pooled = m.reshape(Hb, block, Wb, block).max(axis=(1, 3))
    snapped = np.repeat(np.repeat(pooled, block, axis=0), block, axis=1)
    return snapped[:H, :W].astype(np.float32)


def _widen_mask_rows(mask_u8, y_breaks, ratios, transition_px=None):
    """Per-row horizontal scale of a binary mask about the row centroid.
    y_breaks = (shoulder_y, hip_y, knee_y, ankle_y) in this image;
    ratios = (torso, thigh, shin), applied as plateaus with soft ramps.
    Widen-only: the result is unioned with the input."""
    H, W = mask_u8.shape[:2]
    sh, hp, kn, an = [float(v) for v in y_breaks]
    rt, rth, rsh = [max(float(r), 1.0) for r in ratios]
    if max(rt, rth, rsh) < 1.005:
        return mask_u8
    if transition_px is None:
        transition_px = max(8.0, 0.05 * H)
    t = float(transition_px)
    ys = [sh - t, sh, hp - 1.0, hp, kn - 1.0, kn, an, an + t]
    vs = [1.0, rt, rt, rth, rth, rsh, rsh, 1.0]
    for kk in range(1, len(ys)):
        if ys[kk] <= ys[kk - 1]:
            ys[kk] = ys[kk - 1] + 1e-3
    row_r = np.interp(np.arange(H, dtype=np.float64), ys, vs, left=1.0, right=1.0)
    cols = np.arange(W, dtype=np.float32)
    mf = mask_u8.astype(np.float32)
    row_sum = mf.sum(axis=1)
    cx = np.full(H, W / 2.0, dtype=np.float32)
    nz = row_sum > 0
    if nz.any():
        cx[nz] = (mf[nz] * cols[None, :]).sum(axis=1) / row_sum[nz]
    map_x = (cx[:, None] + (cols[None, :] - cx[:, None])
             / row_r[:, None].astype(np.float32)).astype(np.float32)
    map_y = np.repeat(np.arange(H, dtype=np.float32)[:, None], W, axis=1)
    out = cv2.remap(mask_u8, map_x, map_y, cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.maximum(out, mask_u8)


def _fit_intrinsics(kp3d, cam_t, kp2d):
    """Least-squares pinhole fit from a recon's own 3D->2D correspondences.
    Exactly the projection model the per-frame loop uses; self-consistent with
    the recon's kp2d pixel space by construction."""
    j3d_cam = np.asarray(kp3d, dtype=np.float64) + np.asarray(cam_t, dtype=np.float64)
    xn = j3d_cam[:, 0] / j3d_cam[:, 2]
    yn = j3d_cam[:, 1] / j3d_cam[:, 2]
    Ax = np.stack([xn, np.ones_like(xn)], axis=-1)
    Ay = np.stack([yn, np.ones_like(yn)], axis=-1)
    k2 = np.asarray(kp2d, dtype=np.float64)
    fx, cx = np.linalg.lstsq(Ax, k2[:, 0], rcond=None)[0]
    fy, cy = np.linalg.lstsq(Ay, k2[:, 1], rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _border_color_silhouette(img_bgr, mesh_sil, border=10, thr=30.0):
    """Silhouette for UNIFORM (studio) backgrounds: sample the border ring; if
    it is near-uniform, everything sufficiently far from the border color is
    foreground. Deterministic, no models, no VRAM. Returns mask or None when
    the background is not uniform enough to trust."""
    try:
        H, W = mesh_sil.shape[:2]
        img = img_bgr[..., :3].astype(np.float32)
        ring = np.concatenate([
            img[:border].reshape(-1, 3), img[-border:].reshape(-1, 3),
            img[:, :border].reshape(-1, 3), img[:, -border:].reshape(-1, 3),
        ], axis=0)
        med = np.median(ring, axis=0)
        dev = np.abs(ring - med[None, :]).mean(axis=1)
        # In a knee or full-body crop the subject runs off the bottom edge, so
        # a large slice of the ring IS the person. Judging uniformity on the
        # 90th percentile then rejects a perfectly flat studio background.
        # Judge it on the inlier half instead, and re-take the median from the
        # inliers so the subject cannot drag the background colour.
        _in = dev <= max(np.quantile(dev, 0.5), 1e-6)
        if int(_in.sum()) >= 64:
            med = np.median(ring[_in], axis=0)
            dev = np.abs(ring - med[None, :]).mean(axis=1)
        if float(np.quantile(dev, 0.55)) > 0.6 * thr:
            return None
        dist = np.abs(img - med[None, None, :]).mean(axis=2)
        fg = (dist > thr).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        n, lab = cv2.connectedComponents(fg)
        keep = np.zeros_like(fg)
        for c in range(1, n):
            comp = lab == c
            if (mesh_sil[comp] > 0).any():
                keep[comp] = 1
        return keep if int(keep.sum()) > 200 else None
    except Exception as e:
        print(f"[BetaSwap] clothing: border-color seg failed ({type(e).__name__}: {e})")
        return None


def _grabcut_silhouette(img_bgr, mesh_sil):
    """Clothed-person silhouette via GrabCut seeded by the projected MHR mesh.
    The probable-FG ring is horizontally biased and sized from the LOCAL body
    width (median row width of the mesh), so it cannot swallow arbitrary
    background the way a big isotropic ring can. Returns (silhouette,
    probable_region) or None; the caller validates that GrabCut actually cut
    something before trusting the result."""
    try:
        H, W = mesh_sil.shape[:2]
        row_w = mesh_sil.sum(axis=1)
        widths = row_w[row_w > 4]
        if widths.size < 8:
            return None
        mrw = float(np.median(widths))
        kx = int(min(max(round(mrw), 9), 181))
        if kx % 2 == 0:
            kx += 1
        k_er = max(3, int(round(H * 0.01)))
        sure = cv2.erode(mesh_sil, np.ones((k_er, k_er), dtype=np.uint8))
        prob = cv2.dilate(mesh_sil, np.ones((1, kx), dtype=np.uint8))
        prob = cv2.dilate(prob, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        gc = np.full((H, W), cv2.GC_BGD, dtype=np.uint8)
        gc[prob > 0] = cv2.GC_PR_FGD
        gc[sure > 0] = cv2.GC_FGD
        bgd = np.zeros((1, 65), dtype=np.float64)
        fgd = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(np.ascontiguousarray(img_bgr[..., :3]), gc, None,
                    bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
        n, lab = cv2.connectedComponents(fg)
        keep = np.zeros_like(fg)
        for c in range(1, n):
            comp = lab == c
            if (mesh_sil[comp] > 0).any():
                keep[comp] = 1
        if int(keep.sum()) <= 100:
            return None
        return keep, prob
    except Exception as e:
        print(f"[BetaSwap] clothing: GrabCut failed ({type(e).__name__}: {e})")
        return None


def _align_mesh_to_recon(mesh_verts, mesh_j3d, kp3d_recon):
    """mhr_forward returns the body in the RIG ROOT frame; pred_* live in the
    RECON frame. In the per-frame loop that constant offset is silently eaten
    by the anchor pin (cam_t is re-solved so the anchor lands on the driver
    pixel), but the one-shot measurement has no pin - unaligned, the mesh
    projects far from the skeleton and the sanity gate rejects everything
    (observed: '0/6 joints inside').

    The two frames differ by a pure translation, so the median joint delta
    recovers it exactly. Returns (aligned_verts, delta)."""
    v = np.asarray(mesh_verts, dtype=np.float64).reshape(-1, 3)
    if mesh_j3d is None or kp3d_recon is None:
        return v, np.zeros(3, dtype=np.float64)
    j = np.asarray(mesh_j3d, dtype=np.float64).reshape(-1, 3)
    k = np.asarray(kp3d_recon, dtype=np.float64).reshape(-1, 3)
    n = int(min(j.shape[0], k.shape[0]))
    if n < 8:
        return v, np.zeros(3, dtype=np.float64)
    delta = np.median(k[:n] - j[:n], axis=0)
    if not np.all(np.isfinite(delta)):
        return v, np.zeros(3, dtype=np.float64)
    return v + delta[None, :], delta


_CLOTH_WORK_MAXDIM = 1600.0


_ZONE_RATIO_CAP = 1.45   # a cape read x3.18 on the shin; that is not a leg
_ZONE_LAST = {}          # last photo-vs-photo zone ratios, for the report
_MASK_BUDGET = 1.75      # mask area / body silhouette area; above ~1.9 the model
                         # starts inventing furniture and backdrop


def _measure_clothing_ratios(out_dict, img_bgr, user_mask_np,
                             mesh_verts=None, cam_int=None, mesh_j3d=None):
    """Per-zone (torso / thigh / shin) width ratio between the CLOTHED
    reference silhouette and the projected minimal MHR body mesh, in the
    reference image. Returns {'torso','thigh','shin','raw','source'} or None.

    WORKING RESOLUTION IS CAPPED (1600 px on the long side). The mesh is a
    ~18k point cloud: rasterizing it into a full-resolution reference (a 4608
    x8192 upscale, for instance) puts ~37 px between neighbouring vertices,
    while the morphological close kernel is capped at 31 px - the splat never
    closes into a body, the skeleton joints land in the gaps, and the sanity
    gate rejects a measurement that was geometrically correct all along
    (observed: '0/6 joints inside' on every run). Downscaling first makes the
    point density high enough to close, and every quantity here is a ratio, so
    the result is unchanged. It is also ~25x cheaper.

    Mesh source: mesh_verts + cam_int (mhr_forward output projected with the
    same lstsq-fitted intrinsics the frame loop uses), translated into the
    recon frame by the median joint delta (mesh_j3d vs pred_keypoints_3d).
    Fallback: pred_vertices + focal_length.

    Sanity gate (rejects any broken mesh instead of inflating everyone):
    - in-image skeleton joints must lie inside the dilated mesh silhouette
      (>= 75%);
    - median mesh torso row width must be >= 0.35 x the shoulder pixel span.
    Segmentation priority: user mask > border-color (uniform background) >
    GrabCut (validated). Zones gated by joint visibility; missing lower zones
    inherit half of the torso excess; per-row medians, clamped to
    [0.9, _ZONE_RATIO_CAP]. A cape or a pleated skirt is not a limb: raw shin
    ratios of 3.18 and thigh of 2.24 were measured, and a mask inflated that
    far reads as room for an object Wan then invents."""
    kp2d = out_dict.get("pred_keypoints_2d")
    if kp2d is None or img_bgr is None:
        return None
    H0, W0 = img_bgr.shape[:2]
    kp2d = np.asarray(kp2d, dtype=np.float64)
    sc = min(1.0, _CLOTH_WORK_MAXDIM / float(max(H0, W0)))
    if sc < 1.0:
        H, W = int(round(H0 * sc)), int(round(W0 * sc))
        img_bgr = cv2.resize(img_bgr[..., :3], (W, H), interpolation=cv2.INTER_AREA)
        kp2d = kp2d * sc
        print(f"[BetaSwap] clothing_volume: reference {W0}x{H0} -> working "
              f"{W}x{H} (mesh splat needs vertex density; ratios are unaffected)")
    else:
        H, W = H0, W0
    mesh_sil = None
    align_delta = None
    if mesh_verts is not None and cam_int is not None:
        v, align_delta = _align_mesh_to_recon(
            mesh_verts, mesh_j3d, out_dict.get("pred_keypoints_3d"))
        cam_t = np.asarray(out_dict.get("pred_cam_t"), dtype=np.float64).reshape(-1)[:3]
        v2d = _perspective_project(v + cam_t[None, :], cam_int) * sc
        mesh_sil = _splat_silhouette(v2d, H, W, extra_dilate=0)
    if mesh_sil is None:
        verts = out_dict.get("pred_vertices")
        cam_t = out_dict.get("pred_cam_t")
        focal = out_dict.get("focal_length")
        if verts is None or cam_t is None or focal is None:
            return None
        v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
        v2d = _project_with_focal(
            v + np.asarray(cam_t, dtype=np.float64).reshape(-1)[:3][None, :],
            float(np.asarray(focal).reshape(-1)[0]), W0, H0) * sc
        mesh_sil = _splat_silhouette(v2d, H, W, extra_dilate=0)
    if mesh_sil is None or int(mesh_sil.sum()) < 200:
        return None

    def _in_img(i):
        return (0.0 <= kp2d[i, 0] < W) and (0.0 <= kp2d[i, 1] < H)

    # --- sanity gate: the mesh silhouette must agree with the skeleton ---
    chk = [i for i in (5, 6, 9, 10, 11, 12) if _in_img(i)]
    if len(chk) >= 2:
        sil_dil = cv2.dilate(mesh_sil, np.ones((9, 9), dtype=np.uint8))
        inside = sum(1 for i in chk
                     if sil_dil[int(kp2d[i, 1]), int(kp2d[i, 0])] > 0)
        if inside / len(chk) < 0.75:
            print(f"[BetaSwap] clothing_volume: mesh silhouette disagrees with "
                  f"the skeleton ({inside}/{len(chk)} joints inside) -> "
                  f"measurement rejected, feature off this run.")
            return None
    if _in_img(5) and _in_img(6) and _in_img(9) and _in_img(10):
        sh_span = float(np.linalg.norm(kp2d[5] - kp2d[6]))
        a = int(max(0, min(kp2d[5, 1], kp2d[9, 1])))
        b = int(min(H, max(kp2d[6, 1], kp2d[10, 1])))
        tw = [int(mesh_sil[y].sum()) for y in range(a, b)]
        tw = [w for w in tw if w > 4]
        if sh_span > 8.0 and (len(tw) < 8
                              or float(np.median(tw)) < 0.35 * sh_span):
            print("[BetaSwap] clothing_volume: mesh torso width implausible vs "
                  "shoulder span -> measurement rejected, feature off this run.")
            return None

    grab_prob = None
    if user_mask_np is not None:
        seg = np.asarray(user_mask_np, dtype=np.float32)
        if seg.shape[:2] != (H, W):
            seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
        seg = (seg > 0.5).astype(np.uint8)
        source = "reference_body_mask"
    else:
        seg = _border_color_silhouette(img_bgr, mesh_sil)
        source = "border-color (uniform background)"
        if seg is None:
            _g = _grabcut_silhouette(img_bgr, mesh_sil)
            if _g is None:
                return None
            seg, grab_prob = _g
            source = "auto GrabCut (mesh-seeded)"
    sh_y = float((kp2d[5, 1] + kp2d[6, 1]) / 2.0)
    hp_y = float((kp2d[9, 1] + kp2d[10, 1]) / 2.0)
    kn_y = float((kp2d[11, 1] + kp2d[12, 1]) / 2.0)
    an_y = float((kp2d[13, 1] + kp2d[14, 1]) / 2.0)
    zone_ok = {
        "torso": all(_in_img(i) for i in (5, 6, 9, 10)),
        "thigh": all(_in_img(i) for i in (9, 10, 11, 12)),
        "shin": all(_in_img(i) for i in (11, 12, 13, 14)),
    }
    raw = {}
    for name, (y0, y1) in (("torso", (sh_y, hp_y)),
                           ("thigh", (hp_y, kn_y)),
                           ("shin", (kn_y, an_y))):
        if not zone_ok[name]:
            raw[name] = None
            continue
        a = int(max(0, min(y0, y1)))
        b = int(min(H, max(y0, y1)))
        rs = []
        for y in range(a, b):
            wm = int(mesh_sil[y].sum())
            ws = int(seg[y].sum())
            if wm > 4 and ws > 4:
                rs.append(ws / wm)
        raw[name] = float(np.median(rs)) if len(rs) >= 8 else None
    if raw.get("torso") is None:
        print("[BetaSwap] clothing_volume: torso zone not measurable on the "
              "reference (joints cropped / out of frame) -> disabled this run.")
        return None
    if grab_prob is not None:
        cut_frac = 1.0 - float(seg.sum()) / max(float((grab_prob > 0).sum()), 1.0)
        measured = [raw[z] for z in ("torso", "thigh", "shin") if raw[z] is not None]
        pegged = len(measured) > 0 and all(v >= 1.75 for v in measured)
        if cut_frac < 0.12 or pegged:
            print(f"[BetaSwap] clothing_volume: auto GrabCut unreliable "
                  f"(cut_frac={cut_frac:.2f}, raw={ {k: (round(v, 2) if v else None) for k, v in raw.items()} }) "
                  f"-> disabled this run. Connect reference_body_mask for a "
                  f"real measurement.")
            return None

    def _clamp(v):
        return float(min(max(v, 0.9), _ZONE_RATIO_CAP))

    out = {"torso": _clamp(raw["torso"])}
    inherit = 1.0 + 0.5 * max(out["torso"] - 1.0, 0.0)
    for z in ("thigh", "shin"):
        out[z] = _clamp(raw[z]) if raw[z] is not None else inherit
    out["raw"] = {k: (round(v, 3) if v is not None else None) for k, v in raw.items()}
    out["source"] = source
    out["align"] = (round(float(np.linalg.norm(align_delta)), 4)
                    if align_delta is not None else None)
    out["work"] = f"{W}x{H}"
    return out


# ---------------------------------------------------------------------------
# Diagnostics. Pure measurement: nothing below changes a single output pixel.
# ---------------------------------------------------------------------------

def _umeyama_sim(src, dst):
    """Best-fit similarity (scale, rotation, translation) mapping src -> dst.
    Used to answer one question with a number instead of an opinion: are the
    mhr_forward frame and the recon frame related by a pure TRANSLATION (then
    scale==1, rot==0 and the median-delta alignment is correct), or by
    something else (then it is not)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = int(min(len(src), len(dst)))
    src = src[:n]
    dst = dst[:n]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    s0 = src - mu_s
    d0 = dst - mu_d
    var_s = float((s0 ** 2).sum() / max(n, 1))
    C = (d0.T @ s0) / max(n, 1)
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    scale = float((D * np.diag(S)).sum() / max(var_s, 1e-12))
    t = mu_d - scale * (R @ mu_s)
    res = np.linalg.norm(dst - (scale * (src @ R.T) + t), axis=1)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return {"scale": scale, "R": R, "t": t,
            "res_med": float(np.median(res)), "rot_deg": ang}


def _joints_inside(sil, kp2d, idxs, dilate=9):
    """How many of the given skeleton joints land inside a silhouette - the
    exact quantity the clothing_volume gate rejects on."""
    if sil is None:
        return 0, len(idxs)
    H, W = sil.shape[:2]
    d = cv2.dilate(sil, np.ones((dilate, dilate), dtype=np.uint8))
    ok = 0
    tot = 0
    for i in idxs:
        x, y = float(kp2d[i, 0]), float(kp2d[i, 1])
        if not (0.0 <= x < W and 0.0 <= y < H):
            continue
        tot += 1
        if d[int(y), int(x)] > 0:
            ok += 1
    return ok, tot


def _bbox_str(sil):
    if sil is None:
        return "none"
    ys, xs = np.nonzero(sil)
    if xs.size == 0:
        return "empty"
    return (f"x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}] "
            f"area={int(sil.sum())}")


def _block_flips(a, b, block=32):
    """Number of 32px blocks whose on/off state differs between two masks -
    a direct count of the conditioning flicker Wan sees per frame."""
    if a is None or b is None or a.shape != b.shape:
        return -1
    H, W = a.shape[:2]
    ph, pw = (-H) % block, (-W) % block
    fa = np.pad((a > 0.5).astype(np.uint8), ((0, ph), (0, pw)))
    fb = np.pad((b > 0.5).astype(np.uint8), ((0, ph), (0, pw)))
    Hb, Wb = fa.shape[0] // block, fa.shape[1] // block
    pa = fa.reshape(Hb, block, Wb, block).max(axis=(1, 3))
    pb = fb.reshape(Hb, block, Wb, block).max(axis=(1, 3))
    return int((pa != pb).sum())


def _jerk(x_t, x_t1, x_t2):
    """Median per-point second difference in px: 0 for smooth motion of any
    speed, large for frame-to-frame shake. This is the shakiness metric."""
    if x_t is None or x_t1 is None or x_t2 is None:
        return float("nan")
    a = np.asarray(x_t, dtype=np.float64)
    b = np.asarray(x_t1, dtype=np.float64)
    c = np.asarray(x_t2, dtype=np.float64)
    if a.shape != b.shape or a.shape != c.shape:
        return float("nan")
    return float(np.median(np.linalg.norm(a - 2.0 * b + c, axis=-1)))


def _diag_reference_mesh(out_dict, mesh_j3d, mesh_verts, cam_int, img_bgr,
                         save_path=None):
    """Why does the clothing_volume gate reject? Prints, for every candidate
    mesh->image path, how many skeleton joints actually land inside the
    projected mesh silhouette, plus the frame relation (translation vs
    similarity) between mhr_forward joints and the recon joints."""
    try:
        kp2d = np.asarray(out_dict.get("pred_keypoints_2d"), dtype=np.float64)
        kp3d = np.asarray(out_dict.get("pred_keypoints_3d"), dtype=np.float64)
        cam_t = np.asarray(out_dict.get("pred_cam_t"), dtype=np.float64).reshape(-1)[:3]
        H, W = img_bgr.shape[:2]
        chk = (5, 6, 9, 10, 11, 12)
        print(f"[BetaSwap][DIAG] --- reference mesh report ---")
        print(f"[BetaSwap][DIAG] ref image {W}x{H}; skeleton bbox "
              f"x[{kp2d[:70, 0].min():.0f},{kp2d[:70, 0].max():.0f}] "
              f"y[{kp2d[:70, 1].min():.0f},{kp2d[:70, 1].max():.0f}]; "
              f"cam_t={np.round(cam_t, 4)}")
        print(f"[BetaSwap][DIAG] recon joint cloud: centroid="
              f"{np.round(kp3d[:70].mean(axis=0), 4)}, "
              f"radius_med={float(np.median(np.linalg.norm(kp3d[:70] - kp3d[:70].mean(axis=0), axis=1))):.4f}")

        variants = []
        if mesh_verts is not None and cam_int is not None:
            v = np.asarray(mesh_verts, dtype=np.float64).reshape(-1, 3)
            print(f"[BetaSwap][DIAG] mhr mesh: {v.shape[0]} verts, centroid="
                  f"{np.round(v.mean(axis=0), 4)}, "
                  f"radius_med={float(np.median(np.linalg.norm(v - v.mean(axis=0), axis=1))):.4f}")
            variants.append(("A mhr raw", v, cam_int, cam_t))
            if mesh_j3d is not None:
                j = np.asarray(mesh_j3d, dtype=np.float64).reshape(-1, 3)
                n = int(min(len(j), len(kp3d)))
                d = np.median(kp3d[:n] - j[:n], axis=0)
                res = float(np.median(np.linalg.norm((kp3d[:n] - j[:n]) - d[None, :], axis=1)))
                print(f"[BetaSwap][DIAG] mhr joints vs recon joints: "
                      f"median delta={np.round(d, 4)}, residual_after_translation={res:.4f} "
                      f"(pure translation <=> residual ~0)")
                sim = _umeyama_sim(j[:n], kp3d[:n])
                print(f"[BetaSwap][DIAG] similarity fit: scale={sim['scale']:.4f} "
                      f"rot={sim['rot_deg']:.2f}deg t={np.round(sim['t'], 4)} "
                      f"residual={sim['res_med']:.4f}")
                variants.append(("B mhr+translation", v + d[None, :], cam_int, cam_t))
                variants.append(("C mhr+similarity",
                                 sim["scale"] * (v @ sim["R"].T) + sim["t"][None, :],
                                 cam_int, cam_t))
        verts_l = out_dict.get("pred_vertices")
        focal = out_dict.get("focal_length")

        sc = min(1.0, _CLOTH_WORK_MAXDIM / float(max(H, W)))
        Hs, Ws = int(round(H * sc)), int(round(W * sc))
        print(f"[BetaSwap][DIAG] splat test at full {W}x{H} and at working {Ws}x{Hs} "
              f"(mesh is a point cloud: too few verts per pixel and it never closes "
              f"into a body)")
        sils = {}
        for name, vv, K, ct in variants:
            v2d_full = _perspective_project(vv + ct[None, :], K)
            sil = _splat_silhouette(v2d_full, H, W, extra_dilate=0)
            ok, tot = _joints_inside(sil, kp2d, chk)
            sil_s = _splat_silhouette(v2d_full * sc, Hs, Ws, extra_dilate=0)
            ok_s, tot_s = _joints_inside(sil_s, kp2d * sc, chk)
            fill = (float(sil.sum()) / max(float(np.count_nonzero(
                sil.any(axis=1)) * np.count_nonzero(sil.any(axis=0))), 1.0)
                if sil is not None else 0.0)
            sils[name] = sil
            print(f"[BetaSwap][DIAG] {name:22s} full-res inside {ok}/{tot} "
                  f"(bbox fill {fill:.2%}) | working-res inside {ok_s}/{tot_s}  "
                  f"{_bbox_str(sil)}")
        if verts_l is not None and focal is not None:
            vl = np.asarray(verts_l, dtype=np.float64).reshape(-1, 3)
            f = float(np.asarray(focal).reshape(-1)[0])
            v2d = _project_with_focal(vl + cam_t[None, :], f, W, H)
            sil = _splat_silhouette(v2d, H, W, extra_dilate=0)
            ok, tot = _joints_inside(sil, kp2d, chk)
            sil_s = _splat_silhouette(v2d * sc, Hs, Ws, extra_dilate=0)
            ok_s, tot_s = _joints_inside(sil_s, kp2d * sc, chk)
            sils["D pred_vertices+focal"] = sil
            print(f"[BetaSwap][DIAG] {'D pred_vertices+focal':22s} full-res inside {ok}/{tot} "
                  f"| working-res inside {ok_s}/{tot_s}  {_bbox_str(sil)} (focal={f:.1f})")

        if save_path:
            canvas = img_bgr[..., :3].copy()
            colors = {"A mhr raw": (0, 0, 255), "B mhr+translation": (0, 255, 255),
                      "C mhr+similarity": (0, 255, 0), "D pred_vertices+focal": (255, 0, 255)}
            for name, sil in sils.items():
                if sil is None:
                    continue
                cn, _ = cv2.findContours((sil > 0).astype(np.uint8),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas, cn, -1, colors.get(name, (255, 255, 255)), 2)
            for i in chk:
                p = (int(kp2d[i, 0]), int(kp2d[i, 1]))
                cv2.circle(canvas, p, 6, (255, 255, 255), -1)
                cv2.circle(canvas, p, 6, (0, 0, 0), 2)
            # The reference can be a 4096x8192 upscale; a full-size overlay PNG is
            # ~10 MB of nothing. Nothing is measured off the picture, it is a human
            # sanity check only, so cap the long side.
            _oc = min(1.0, 900.0 / float(max(canvas.shape[:2])))
            if _oc < 1.0:
                canvas = cv2.resize(
                    canvas, (max(int(round(canvas.shape[1] * _oc)), 1),
                             max(int(round(canvas.shape[0] * _oc)), 1)),
                    interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".png", canvas)
            if ok:
                buf.tofile(save_path)
                print(f"[BetaSwap][DIAG] overlay saved: {save_path} "
                      f"(red=raw, yellow=translation, green=similarity, magenta=legacy, "
                      f"white dots=joints the gate checks)")
    except Exception as e:
        import traceback
        print(f"[BetaSwap][DIAG] reference mesh report failed: "
              f"{type(e).__name__}: {e}\n{traceback.format_exc()}")



class _StdoutTee:
    """Mirrors everything the node prints into the single report file, so the
    console transcript never has to be saved separately."""

    def __init__(self, sink):
        self.sink = sink
        self.orig = None

    def write(self, text):
        if self.orig is not None:
            self.orig.write(text)
        try:
            self.sink.append(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        if self.orig is not None:
            self.orig.flush()

    def __enter__(self):
        self.orig = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *a):
        sys.stdout = self.orig
        self.orig = None
        return False


class _DiagLog:
    """Collects the full DIAG+ report in memory and writes it as ONE text file.
    Console keeps only the short lines; everything bulky (45-long shape vectors,
    the per-index beta probe table, the reference framing report) goes to the
    file so the console log stays readable."""

    def __init__(self, path=None):
        self.path = path
        self.buf = []
        self.console = []
        self.rows = []

    def append(self, text):
        self.console.append(text)

    def __call__(self, line="", echo=False):
        self.buf.append(str(line))
        if echo:
            print(line)

    def table(self, rows):
        """Per-frame values as a csv block inside the same file - no side .csv."""
        if not rows:
            return
        cols = []
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        self("")
        self("===== per-frame table (csv block) =====")
        self(",".join(cols))
        for r in rows:
            self(",".join(str(r.get(c, "")) for c in cols))

    def flush(self):
        if not self.path:
            return
        try:
            body = ("".join(self.console)).replace("\r\n", "\n").rstrip("\n")
            with open(self.path, "w", encoding="utf-8", newline="\r\n") as f:
                f.write("===== console transcript =====\n")
                f.write(body + "\n\n")
                f.write("\n".join(self.buf) + "\n")
            print(f"[BetaSwap][DIAG+] EVERYTHING is in this one file -> {self.path}")
        except Exception as e:
            print(f"[BetaSwap][DIAG+] could not write the report file: {e}")


def _vec(x):
    """Any torch/numpy param -> flat float64 numpy."""
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _mhr_verts_for_shape(sam_3d_model, out_dict, device, shape_vec,
                         dtype=torch.float32):
    """mhr_forward on out_dict's OWN pose with an arbitrary shape vector.
    Same tensor prep and same axis flip as _beta_swap_forward, so the geometry
    that comes back is directly comparable to what the node actually renders."""

    def _to_t(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return (x.to(device=device, dtype=dtype).unsqueeze(0)
                if x.ndim == 1 else x.to(device=device, dtype=dtype))

    global_rot = _to_t(out_dict["global_rot"])
    body_pose_params = _to_t(out_dict["body_pose_params"])[:, :130]
    hand_pose_params = _to_t(out_dict["hand_pose_params"])
    expr_params = _to_t(out_dict.get("expr_params"))
    scale_params = _to_t(out_dict["scale_params"])
    sp = _to_t(np.asarray(shape_vec, dtype=np.float32))
    with torch.no_grad():
        out = sam_3d_model.head_pose.mhr_forward(
            global_trans=torch.zeros_like(global_rot),
            global_rot=global_rot,
            body_pose_params=body_pose_params,
            hand_pose_params=hand_pose_params,
            scale_params=scale_params,
            shape_params=sp,
            expr_params=expr_params,
            return_keypoints=True,
            do_pcblend=True,
        )
    if not isinstance(out, (tuple, list)):
        raise RuntimeError("mhr_forward returned unexpected single tensor")
    v = out[0].detach().cpu().numpy()[0].copy()
    v[..., [1, 2]] *= -1
    j = out[1].detach().cpu().numpy()[0][:70].copy()
    j[..., [1, 2]] *= -1
    return v, j


def _mesh_shape_metrics(verts, j3d=None, nslab=48):
    """Geometry of a posed MHR mesh, in metres, with no reference to any beta.
    vol is an elliptical-slab volume proxy (sum of pi/4 * width * depth * dh):
    it is the closest thing to 'body mass' that can be read off the mesh
    itself, which is the only ground truth available here."""
    v = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    y = v[:, 1]
    y0, y1 = float(y.min()), float(y.max())
    h = max(y1 - y0, 1e-6)
    head_at_min = True
    if j3d is not None and len(np.asarray(j3d)):
        ny = float(np.asarray(j3d, dtype=np.float64).reshape(-1, 3)[0, 1])
        head_at_min = abs(ny - y0) < abs(ny - y1)
    edges = np.linspace(y0, y1, nslab + 1)
    idx = np.clip(np.digitize(y, edges) - 1, 0, nslab - 1)
    w = np.zeros(nslab, dtype=np.float64)
    d = np.zeros(nslab, dtype=np.float64)
    for k in range(nslab):
        m = idx == k
        if int(m.sum()) >= 8:
            w[k] = float(v[m, 0].max() - v[m, 0].min())
            d[k] = float(v[m, 2].max() - v[m, 2].min())
    vol = float((np.pi / 4.0) * float((w * d).sum()) * (h / nslab))
    f = (np.arange(nslab, dtype=np.float64) + 0.5) / nslab
    if not head_at_min:
        f = 1.0 - f

    def _band(a, b):
        m = (f >= a) & (f < b) & (w > 0)
        return float(np.median(w[m])) if bool(m.any()) else 0.0

    return {"height": h, "vol": vol,
            "head": _band(0.00, 0.13), "chest": _band(0.20, 0.30),
            "waist": _band(0.33, 0.43), "hip": _band(0.45, 0.55),
            "thigh": _band(0.58, 0.70), "shin": _band(0.75, 0.88)}


def _build_ratio(sam_3d_model, pose_out, device, ref_shape, dri_shape, log=None):
    """How much bulkier the reference body is than the driver body, measured on
    the MESHES, not on a beta.

    The probe run proved index 1 is not mass: across four references its dVol was
    -0.0008/-0.0005/-0.0002/+0.0001 - noise level, sign flipping - while index 0
    carried +0.012..+0.017 every time. But hard-coding index 0 would just be the
    next guess: MHR betas are PCA weights, so the ordering is a property of the
    basis and can move with the checkpoint. So read no beta at all. Pose BOTH
    shape vectors identically, measure the two meshes, and divide.

    Girth is normalised by height, so a taller reference is not also read as a
    fatter one (height is already handled by auto_height/s_eff)."""
    try:
        v_r, j_r = _mhr_verts_for_shape(sam_3d_model, pose_out, device, ref_shape)
        v_d, j_d = _mhr_verts_for_shape(sam_3d_model, pose_out, device, dri_shape)
        m_r = _mesh_shape_metrics(v_r, j_r)
        m_d = _mesh_shape_metrics(v_d, j_d)

        def _g(m):
            w = [m[k] for k in ("chest", "waist", "thigh") if m[k] > 1e-6]
            return (float(np.mean(w)) / max(m["height"], 1e-6)) if w else None

        g_r, g_d = _g(m_r), _g(m_d)
        if not g_r or not g_d:
            return 1.0, None
        ratio = float(g_r / g_d)
        det = {"ref": m_r, "dri": m_d, "g_ref": g_r, "g_dri": g_d, "ratio": ratio,
               "vol_ratio": float(m_r["vol"] / max(m_d["vol"], 1e-9))}
        if log is not None:
            log("")
            log("===== build ratio (mesh measured, no beta involved) =====")
            for nm, m in (("reference", m_r), ("driver", m_d)):
                log(f"  {nm:9s} height={m['height']:.4f} vol={m['vol']:.6f} "
                    f"chest={m['chest']:.4f} waist={m['waist']:.4f} "
                    f"hip={m['hip']:.4f} thigh={m['thigh']:.4f}")
            log(f"  girth/height: ref={g_r:.5f} driver={g_d:.5f} -> build ratio "
                f"{ratio:.4f} (raw volume ratio {det['vol_ratio']:.4f})")
        return ratio, det
    except Exception as e:
        if log is not None:
            log(f"build ratio failed: {type(e).__name__}: {e}")
        return 1.0, None


def _measure_headwear(img_bgr, mesh_sil, kp2d, log=None, label=""):
    """How much room whatever is worn on the head needs, in units of the bare
    mesh head width.

    The MHR mesh is bald and earless, so everything that matters here - hair,
    headband, bunny ears, a hat - lives strictly ABOVE and AROUND the mesh crown
    and is invisible to every other measurement in this node. Returns
    (rise, half_width) as multiples of the ear span - the one length that can
    be measured identically on the reference and on every output frame."""
    try:
        H, W = mesh_sil.shape[:2]
        rows = np.nonzero(np.any(mesh_sil > 0, axis=1))[0]
        if not rows.size:
            return 0.0, 0.0
        crown = int(rows.min())
        span = max(int(0.14 * (int(rows.max()) - crown)), 12)
        ws = []
        for y in range(crown, min(crown + span, H)):
            c = np.nonzero(mesh_sil[y] > 0)[0]
            if c.size > 1:
                ws.append(int(c.max() - c.min()))
        head_w = float(np.percentile(ws, 90)) if ws else 0.0
        if head_w < 4.0:
            return 0.0, 0.0
        cx = float(kp2d[0][0]) if kp2d is not None else float(W) / 2.0
        unit = head_w
        unit_name = "mesh head width"
        try:
            _es = float(np.linalg.norm(np.asarray(kp2d[3], dtype=np.float64)
                                       - np.asarray(kp2d[4], dtype=np.float64)))
            if np.isfinite(_es) and _es > 4.0:
                unit, unit_name = _es, "ear span"
        except Exception:
            pass
        # A GLOBAL background model fails on references whose backdrop is only
        # locally clean, and the mesh-seeded GrabCut fallback deletes exactly
        # what has to be found - the mesh is bald, so pale ears above it read as
        # background (bunny ears came back 0.04 ear-spans and got eaten).
        # Model the backdrop from the corners of the head band instead: that
        # patch is nearly always plain even when the rest of the photo is not.
        seg = None
        src = ""
        try:
            _bh = int(min(H, max(crown, 4)))
            _bw = int(max(0.10 * W, 8))
            _corners = np.concatenate([
                img_bgr[:_bh, :_bw].reshape(-1, 3),
                img_bgr[:_bh, W - _bw:].reshape(-1, 3)], axis=0).astype(np.float64)
            if _corners.shape[0] >= 64:
                _bg = np.median(_corners, axis=0)
                _sp = float(np.median(np.abs(_corners - _bg[None, :]).mean(axis=1)))
                if _sp < 14.0:
                    _d = np.abs(img_bgr[..., :3].astype(np.float64)
                                - _bg[None, None, :]).mean(axis=2)
                    seg = (_d > max(4.0 * _sp, 12.0)).astype(np.uint8)
                    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN,
                                           np.ones((3, 3), np.uint8))
                    src = f"local backdrop above the head (spread {_sp:.1f})"
        except Exception:
            seg = None
        if seg is None:
            seg = _border_color_silhouette(img_bgr, mesh_sil)
            src = "flat-background colour distance"
        if seg is None:
            g = _grabcut_silhouette(img_bgr, mesh_sil)
            if g is None:
                if log is not None:
                    log(f"headwear {label}: no silhouette, reserving nothing")
                return 0.0, 0.0
            seg, _ = g
            src = "GrabCut (may clip pale accessories)"
        # only what is above the crown, in a generous column band around the head
        band = int(max(2.5 * head_w, 0.10 * W))
        x0, x1 = int(max(0, cx - band)), int(min(W, cx + band))
        top = seg[:crown, x0:x1] > 0
        if not top.any():
            if log is not None:
                log(f"headwear {label}: nothing above the crown (source: {src})")
            return 0.0, 0.0
        # Only what is ATTACHED to the head counts. Without this any speck of
        # backdrop noise in the band set the reserve: a man with slicked-back
        # hair measured 1.12 ear-spans and got 180px of mask opened over an
        # empty head. Keep the components that touch the crown row.
        _lab_n, _lab = cv2.connectedComponents(
            np.ascontiguousarray(top.astype(np.uint8)), 8)
        _touch = set(np.unique(_lab[max(0, crown - 1 - 0):crown, :]).tolist()) \
            if crown >= 1 else set()
        _touch.discard(0)
        if not _touch:
            _touch = set(np.unique(_lab[crown - 1:crown, :]).tolist()) - {0}
        if not _touch:
            if log is not None:
                log(f"headwear {label}: nothing attached to the head "
                    f"(source: {src}), reserving nothing")
            return 0.0, 0.0
        keep = np.isin(_lab, list(_touch))
        ys, xs = np.nonzero(keep)
        if not ys.size:
            return 0.0, 0.0
        rise = float(crown - int(ys.min()))
        half = float(max(abs(int(xs.max()) + x0 - cx), abs(cx - int(xs.min()) - x0)))
        r_rise, r_half = rise / unit, half / unit
        if log is not None:
            log(f"headwear {label}: source {src} | unit = {unit_name} {unit:.0f}px | "
                f"rises {rise:.0f}px = {r_rise:.2f} units above the crown, reaches "
                f"{half:.0f}px = {r_half:.2f} units sideways")
        # Hard ceiling. Real bunny ears measured 2.74 ear-spans; honouring that
        # opened 400px of mask over the head and the model filled it with the
        # reference backdrop. Better a clipped ear than an invented room.
        return float(min(r_rise, 1.5)), float(min(r_half, 1.0))
    except Exception as e:
        if log is not None:
            log(f"headwear {label} failed: {type(e).__name__}: {e}")
        return 0.0, 0.0

def _ref_photo_pack(out_dict, img_bgr, sam_3d_model, device, label, log=None):
    """Everything that can be READ OFF a reference photograph, in one place:
    the bare mesh silhouette, the clothed silhouette, and the 2D skeleton, all
    at one working resolution.

    Returns None if the clothed silhouette cannot be trusted."""
    try:
        j3d, verts = _beta_swap_forward(
            sam_3d_model, out_dict, out_dict, device,
            shape_strength=0.0, scale_strength=0.0,
            amplify_reference=1.0, ref_body_out=None)
        cam_int = _fit_intrinsics(out_dict["pred_keypoints_3d"],
                                  out_dict["pred_cam_t"],
                                  out_dict["pred_keypoints_2d"])
        v, _ = _align_mesh_to_recon(verts, j3d, out_dict["pred_keypoints_3d"])
        ct = np.asarray(out_dict["pred_cam_t"], dtype=np.float64).reshape(-1)[:3]
        H0, W0 = img_bgr.shape[:2]
        sc = min(1.0, _CLOTH_WORK_MAXDIM / float(max(H0, W0)))
        Hw, Ww = int(round(H0 * sc)), int(round(W0 * sc))
        img = (cv2.resize(img_bgr[..., :3], (Ww, Hw), interpolation=cv2.INTER_AREA)
               if sc < 1.0 else img_bgr[..., :3])
        v2d = _perspective_project(v + ct[None, :], cam_int) * sc
        mesh_sil = _splat_silhouette(v2d, Hw, Ww, extra_dilate=0)
        seg = _border_color_silhouette(img, mesh_sil)
        src = "border-color (uniform background)"
        if seg is None:
            g = _grabcut_silhouette(img, mesh_sil)
            if g is None:
                return None
            seg, _ = g
            src = "GrabCut (mesh-seeded, clips pale accessories)"
        kp = np.asarray(out_dict["pred_keypoints_2d"], dtype=np.float64) * sc
        pack = {"img": img, "mesh_sil": mesh_sil, "sil": seg, "kp": kp,
                "sc": sc, "H": Hw, "W": Ww, "src": src, "label": label,
                "touches_top": bool(np.any(seg[0] > 0))}
        if log is not None:
            log(f"reference pack [{label}]: {Ww}x{Hw} work, silhouette source {src}, "
                f"touches top edge: {pack['touches_top']}")
        return pack
    except Exception as e:
        if log is not None:
            log(f"reference pack [{label}] failed: {type(e).__name__}: {e}")
        return None


def _silhouette_profile(sil, kp2d, label="", log=None):
    """Body widths read off a photograph, in HEAD WIDTHS.

    The unit matters more than anything else here. Normalising by torso length
    (previous build) mixed a vertical length with horizontal ones, so any camera
    tilt foreshortened the denominator only: the driver clip, shot from below on
    a wide lens, read torso=320px and body width=465px, i.e. 1.45 - a number that
    describes the lens, not the man. Both characters then came out at 0.73-0.79
    "slimmer than the driver" and identical to each other.

    Head width is horizontal, so it foreshortens exactly like the widths it
    divides, and it is the least build-dependent length on a body - which is why
    "shoulders are N heads wide" is the classic way to describe a build.

    Legs are measured as the two contiguous runs that contain each knee/ankle,
    not as the full row: a cape or a hanging panel between or beside the legs is
    not a leg, and reading it as one inflates the mask into a shape the video
    model then fills with an invented object."""
    try:
        H, W = sil.shape[:2]
        kp = np.asarray(kp2d, dtype=np.float64)
        sh = (kp[5] + kp[6]) / 2.0
        hp = (kp[_LHIP_IDX] + kp[_RHIP_IDX]) / 2.0
        torso = float(np.linalg.norm(sh - hp))
        head = float(np.linalg.norm(kp[3] - kp[4]))
        if not np.isfinite(head) or head < 6.0:
            rows = np.nonzero(np.any(sil > 0, axis=1))[0]
            if not rows.size:
                return None
            ws = [np.nonzero(sil[y] > 0)[0] for y in
                  range(int(rows.min()), min(int(rows.min()) + 40, H))]
            ws = [float(c.max() - c.min()) for c in ws if c.size > 1]
            head = float(np.median(ws)) * 0.85 if ws else 0.0
        if not np.isfinite(head) or head < 6.0 or torso < 8.0:
            return None

        def _full(y, hb):
            y0, y1 = int(max(0, y - hb)), int(min(H, y + hb + 1))
            ws = []
            for r in range(y0, y1):
                c = np.nonzero(sil[r] > 0)[0]
                if c.size > 1:
                    ws.append(float(c.max() - c.min()))
            return float(np.median(ws)) if ws else None

        def _runs(y, hb, xs):
            """Sum of the contiguous foreground runs containing each x in xs."""
            y0, y1 = int(max(0, y - hb)), int(min(H, y + hb + 1))
            tot = []
            for r in range(y0, y1):
                row = sil[r] > 0
                if not row.any():
                    continue
                acc = 0.0
                seen = []
                for x in xs:
                    xi = int(round(x))
                    if xi < 0 or xi >= W or not row[xi]:
                        continue
                    lo = xi
                    while lo > 0 and row[lo - 1]:
                        lo -= 1
                    hi = xi
                    while hi < W - 1 and row[hi + 1]:
                        hi += 1
                    if (lo, hi) in seen:
                        continue
                    seen.append((lo, hi))
                    acc += float(hi - lo + 1)
                if acc > 0:
                    tot.append(acc)
            return float(np.median(tot)) if tot else None

        kn = (kp[11] + kp[12]) / 2.0
        an = (kp[13] + kp[14]) / 2.0
        hb = int(max(0.06 * torso, 2))
        prof = {}
        for name, y in (("chest", sh[1] + 0.22 * torso),
                        ("waist", sh[1] + 0.62 * torso),
                        ("hip", hp[1] + 0.06 * torso)):
            if np.isfinite(y) and 0 <= y < H:
                w = _full(y, hb)
                if w and w > 2.0:
                    prof[name] = w / head
        for name, y, xs in (
                ("thigh", (hp[1] + kn[1]) / 2.0,
                 ((kp[_LHIP_IDX][0] + kp[11][0]) / 2.0,
                  (kp[_RHIP_IDX][0] + kp[12][0]) / 2.0)),
                ("shin", (kn[1] + an[1]) / 2.0,
                 ((kp[11][0] + kp[13][0]) / 2.0,
                  (kp[12][0] + kp[14][0]) / 2.0))):
            if np.isfinite(y) and 0 <= y < H:
                w = _runs(y, hb, xs)
                if w and w > 2.0:
                    prof[name] = w / head
        if "chest" not in prof and "waist" not in prof:
            return None
        prof["_head_px"] = head
        prof["_torso_heads"] = torso / head
        if log is not None:
            log(f"silhouette profile [{label}]: head {head:.0f}px, torso "
                f"{torso / head:.2f} heads | widths in head-units: "
                + " ".join(f"{k}={v:.3f}" for k, v in prof.items() if k[0] != "_"))
        return prof
    except Exception as e:
        if log is not None:
            log(f"silhouette profile [{label}] failed: {type(e).__name__}: {e}")
        return None

def _validate_profile(prof, mesh_prof, label="", log=None):
    """Throw away zones where the clothed silhouette cannot be the same person.

    Every catastrophic run so far came from an unchecked silhouette, not from
    the arithmetic. A school-uniform reference measured chest 8.44 head-widths
    and shin 10.25 - background, not a girl - and drove the shoulder spread to
    its ceiling. A hoodie reference measured chest 4.29 while the same image
    made clothing_volume report a silhouette FOUR TIMES NARROWER than the mesh.
    A minimal-clothing reference reported a waist twice as wide as its chest.

    clothing_volume already refuses this kind of input; the profile did not.
    The bare projected mesh is the ground truth that is always available: a
    dressed body may be up to ~60% wider than its own mesh and should never be
    much narrower. Anything outside that is not clothing."""
    if not prof:
        return None
    out, dropped = {}, []
    for k, v in prof.items():
        if k.startswith("_"):
            out[k] = v
            continue
        m = (mesh_prof or {}).get(k)
        if m and m > 1e-6:
            r = v / m
            if not (0.75 <= r <= 1.75):
                dropped.append(f"{k}({v:.2f}, x{r:.2f} of mesh)")
                continue
        if k in ("chest", "waist", "hip") and not (1.6 <= v <= 4.4):
            dropped.append(f"{k}({v:.2f} heads, outside 1.6-4.4)")
            continue
        out[k] = v
    ch, wa = out.get("chest"), out.get("waist")
    if ch and wa and (max(ch, wa) / max(min(ch, wa), 1e-6)) > 1.45:
        dropped.append(f"chest/waist disagree ({ch:.2f} vs {wa:.2f})")
        out.pop("waist", None)
    if log is not None and dropped:
        log(f"profile [{label}] rejected zones: " + ", ".join(dropped))
    if not any(k in out for k in ("chest", "waist")):
        if log is not None:
            log(f"profile [{label}] has no trustworthy torso zone -> photo route "
                f"disabled for this run")
        return None
    return out


def _photo_build_ratio(ref_prof, dri_prof, log=None):
    """Reference build vs driver build, both measured on photographs.

    The mesh route under-reads this badly: MHR regresses a clothed photo toward
    an average body, so a superhero in a catsuit and a slim man in a t-shirt came
    out 1.02 apart - which is why nothing ever got thicker. Silhouettes do not
    regress."""
    if not ref_prof or not dri_prof:
        return None, None
    zones = {}
    for k in ("chest", "waist", "hip", "thigh", "shin"):
        a, b = ref_prof.get(k), dri_prof.get(k)
        if a and b and b > 1e-6:
            zones[k] = float(min(max(a / b, 0.55), 1.9))
    if not zones:
        return None, None
    core = ([zones[k] for k in ("chest", "waist") if k in zones]
            or [zones[k] for k in ("hip",) if k in zones]
            or list(zones.values()))
    girth = float(np.mean(core))
    if log is not None:
        log("")
        log("===== build from photographs (reference vs driver) =====")
        log("  per zone ref/driver: "
            + " ".join(f"{k}={v:.3f}" for k, v in zones.items()))
        log("  girth uses chest+waist only. hip/thigh/shin are printed for "
            "inspection but excluded: the arms hang there, and a cape or a "
            "flared skirt reads as body. hip measured 4.1-4.5 head-widths on "
            "the references against 2.3 on the driver - a ratio of ~1.8 that "
            "describes a garment, not a build.")
        log(f"  torso girth ratio = {girth:.4f}  (>1 the character is bulkier "
            f"than the driver, <1 slimmer)")
        _th_r, _th_d = ref_prof.get("_torso_heads"), dri_prof.get("_torso_heads")
        if _th_r and _th_d:
            log(f"  stature: reference torso is {_th_r:.2f} heads, driver {_th_d:.2f} "
                f"-> {_th_r / _th_d:.3f}. This is the only proportion in the photos "
                f"that speaks to how BIG the character is rather than how wide; "
                f"auto_height already carries it.")
    return girth, zones

def _diag_beta_probe(sam_3d_model, out_dict, device, label, log,
                     step=1.0, max_idx=64, fat_idx=1):
    """WHICH shape coefficient carries body mass, and with WHAT SIGN.

    MHR shape params are PCA weights (45 = 20 body + 20 head + 5 hand); there
    is no documented 'fatness' axis and the sign of a PCA component is set by
    the basis, not by anatomy. _read_fat() taking index 1 is an SMPL habit.
    So: perturb every index by +/-step, re-run mhr_forward on the SAME pose,
    and let the resulting mesh say which coefficient moves volume and which
    way. Measurement only - nothing here touches an output pixel."""
    import time as _t
    t0 = _t.time()
    sp = _vec(out_dict.get("shape_params"))
    if sp is None:
        log(f"[beta probe {label}] no shape_params")
        return None
    n = int(min(sp.shape[0], max_idx))
    v0, j0 = _mhr_verts_for_shape(sam_3d_model, out_dict, device, sp)
    m0 = _mesh_shape_metrics(v0, j0)
    log("")
    log(f"===== beta probe: {label} =====")
    log(f"shape_params length = {sp.shape[0]}  (MHR doc: 45 = 20 body + 20 head + 5 hand)")
    log("full shape vector:")
    log("  " + " ".join(f"{i}:{val:+.4f}" for i, val in enumerate(sp)))
    log(f"baseline mesh: height={m0['height']:.4f} vol={m0['vol']:.6f} "
        f"head={m0['head']:.4f} chest={m0['chest']:.4f} waist={m0['waist']:.4f} "
        f"hip={m0['hip']:.4f} thigh={m0['thigh']:.4f} shin={m0['shin']:.4f}")
    log(f"per-index central difference, step=+/-{step} (d/dbeta):")
    log(f"{'i':>3} {'dVol':>12} {'dHeight':>10} {'dChest':>10} {'dWaist':>10} "
        f"{'dHip':>10} {'dThigh':>10} {'dHead':>10}")
    rows = []
    for i in range(n):
        try:
            a = sp.copy(); a[i] += step
            b = sp.copy(); b[i] -= step
            va, ja = _mhr_verts_for_shape(sam_3d_model, out_dict, device, a)
            vb, jb = _mhr_verts_for_shape(sam_3d_model, out_dict, device, b)
            ma = _mesh_shape_metrics(va, ja)
            mb = _mesh_shape_metrics(vb, jb)
            r = {k: (ma[k] - mb[k]) / (2.0 * step) for k in m0}
            r["i"] = i
            rows.append(r)
            log(f"{i:>3} {r['vol']:>12.6f} {r['height']:>10.5f} {r['chest']:>10.5f} "
                f"{r['waist']:>10.5f} {r['hip']:>10.5f} {r['thigh']:>10.5f} "
                f"{r['head']:>10.5f}")
        except Exception as e:
            log(f"{i:>3} probe failed: {type(e).__name__}: {e}")
    if not rows:
        return None
    order = sorted(rows, key=lambda r: -abs(r["vol"]))
    top = order[:8]
    log("ranked by |dVol|: " + ", ".join(
        f"#{r['i']}({r['vol']:+.5f})" for r in top))
    cur = next((r for r in rows if r["i"] == fat_idx), None)
    if cur is not None:
        rank = [r["i"] for r in order].index(fat_idx) + 1
        log(f"index {fat_idx} (the one the old build read as 'fat'): "
            f"dVol={cur['vol']:+.6f}, dWaist={cur['waist']:+.6f}, "
            f"rank {rank}/{len(rows)} by |dVol|.")
        log("  Kept as a watchdog only - the build factor is now measured on the "
            "meshes and reads no index at all. If #0 ever stops leading this "
            "table, the basis moved and something upstream changed.")
    log(f"beta probe took {_t.time() - t0:.1f}s ({2 * n} mhr_forward calls)")
    best = order[0]
    print(f"[BetaSwap][DIAG+] beta probe {label}: mass axis = index {best['i']} "
          f"(dVol={best['vol']:+.5f}); _read_fat uses index {fat_idx} "
          f"(dVol={cur['vol']:+.5f} rank {rank}/{len(rows)})"
          if cur is not None else
          f"[BetaSwap][DIAG+] beta probe {label}: mass axis = index {best['i']}")
    return rows


def _diag_ref_framing(out_dict, img_bgr, mesh_verts, mesh_j3d, cam_int,
                      label, log):
    """How this reference photo is CROPPED, and how much room the photo itself
    gives to whatever sits on top of the head (headband, ears, hat).

    above_head is the number that matters for the accessory question: the gap,
    in head-heights, between the top of the CLOTHED silhouette and the crown of
    the bare MHR mesh. A rabbit-ear headband lives exactly in that gap; a bald
    full-body reference has almost none."""
    try:
        kp2d = np.asarray(out_dict.get("pred_keypoints_2d"), dtype=np.float64)
        H0, W0 = img_bgr.shape[:2]
        log("")
        log(f"===== reference framing: {label} =====")
        log(f"image {W0}x{H0}")
        names = {0: "nose", 3: "ear_l", 4: "ear_r", 5: "sh_l", 6: "sh_r",
                 _LHIP_IDX: "hip_l", _RHIP_IDX: "hip_r", 11: "knee_l",
                 12: "knee_r", 13: "ankle_l", 14: "ankle_r"}

        def _inim(i):
            return bool(0.0 <= kp2d[i, 0] < W0 and 0.0 <= kp2d[i, 1] < H0)

        vis = {v: _inim(k) for k, v in names.items()}
        log("joints inside the image: " + ", ".join(
            f"{k}={int(v)}" for k, v in vis.items()))
        if vis["ankle_l"] and vis["ankle_r"]:
            crop = "full body"
        elif vis["knee_l"] and vis["knee_r"]:
            crop = "knee crop"
        elif vis["hip_l"] and vis["hip_r"]:
            crop = "waist crop"
        else:
            crop = "head / chest crop"
        ear = float(np.linalg.norm(kp2d[3] - kp2d[4]))
        eye_mid = (kp2d[1] + kp2d[2]) / 2.0
        head_h = max(abs(float(kp2d[0, 1] - eye_mid[1])) * 3.2, ear * 1.3, 1.0)
        y0, y1 = float(kp2d[:70, 1].min()), float(kp2d[:70, 1].max())
        body_h = max(y1 - y0, 1.0)
        log(f"crop class: {crop}")
        log(f"ear span {ear:.0f} px | head height (est) {head_h:.0f} px | "
            f"skeleton bbox height {body_h:.0f} px")
        log(f"head / body = {head_h / body_h:.3f} | head / image height = "
            f"{head_h / float(H0):.4f}  (bigger = more pixels per head detail)")
        sc = min(1.0, _CLOTH_WORK_MAXDIM / float(max(H0, W0)))
        Hw, Ww = int(round(H0 * sc)), int(round(W0 * sc))
        img_w = (cv2.resize(img_bgr[..., :3], (Ww, Hw),
                            interpolation=cv2.INTER_AREA) if sc < 1.0
                 else img_bgr[..., :3])
        kw = kp2d * sc
        mesh_sil = None
        if mesh_verts is not None and cam_int is not None:
            v, _ = _align_mesh_to_recon(mesh_verts, mesh_j3d,
                                        out_dict.get("pred_keypoints_3d"))
            ct = np.asarray(out_dict.get("pred_cam_t"),
                            dtype=np.float64).reshape(-1)[:3]
            v2d = _perspective_project(v + ct[None, :], cam_int) * sc
            mesh_sil = _splat_silhouette(v2d, Hw, Ww, extra_dilate=0)
        if mesh_sil is None:
            log("no mesh silhouette -> above-head room not measurable")
            return
        seg = _border_color_silhouette(img_w, mesh_sil)
        src = "border-color (uniform background)"
        if seg is None:
            g = _grabcut_silhouette(img_w, mesh_sil)
            if g is None:
                log("no clothed silhouette -> above-head room not measurable")
                return
            seg, _ = g
            src = "auto GrabCut (mesh-seeded)"
        hw = max(ear * sc * 1.2, 0.06 * Ww)
        c0 = int(max(0, kw[0, 0] - hw))
        c1 = int(min(Ww, kw[0, 0] + hw + 1))
        rm = np.any(mesh_sil[:, c0:c1] > 0, axis=1)
        rs = np.any(seg[:, c0:c1] > 0, axis=1)
        if not rm.any() or not rs.any():
            log("head band empty -> above-head room not measurable")
            return
        crown_mesh = int(np.argmax(rm))
        top_cloth = int(np.argmax(rs))
        above = (crown_mesh - top_cloth) / max(sc, 1e-6)
        log(f"clothed silhouette source: {src} (work {Ww}x{Hw})")
        log(f"bare-mesh crown y={crown_mesh / max(sc, 1e-6):.0f} | clothed top "
            f"y={top_cloth / max(sc, 1e-6):.0f} | ABOVE-HEAD = {above:.0f} px = "
            f"{above / head_h:.2f} head-heights")
        log("  (hair + headband + ears live in ABOVE-HEAD. Near 0 = the photo "
            "has nothing on top of the head to transfer.)")
        print(f"[BetaSwap][DIAG+] ref framing {label}: {crop}, head/image="
              f"{head_h / float(H0):.4f}, above-head={above:.0f}px "
              f"({above / head_h:.2f} head-heights)")
    except Exception as e:
        import traceback as _tb
        log(f"reference framing {label} failed: {type(e).__name__}: {e}")
        log(_tb.format_exc())

def _diag_write_report(rows, csv_path=None, log=None):
    """Write the per-frame CSV and print the summary. Measurement only."""
    try:
        import csv as _csv
        cols = ["frame", "ratio_raw", "s_eff", "chain", "anchor", "stick_w",
                "dri_step", "dri_jerk", "swap_raw_jerk", "swap_out_jerk",
                "off_med", "warp", "sil", "sil_area", "clear", "clear_up_px",
                "mask_area", "blocks_flip_pre", "blocks_flip_post",
                "hand_l", "hand_r", "face_sx", "face_sy",
                "hf_recon", "hf_raw", "hf_vit", "n_prompts", "prompt_churn",
                "mask_frac", "mask_top", "head_top", "headroom_px", "headroom_blk",
                # --- DIAG+ ---
                "ref_fat", "dri_fat", "vol_factor",
                "ear_pre", "ear_post", "ear_draw", "head_w_mesh",
                "sh_pre", "sh_draw", "hip_draw",
                "crown_est", "headroom_pose_px",
                "pose_x0", "pose_x1", "pose_y0", "pose_y1",
                "pose_w", "pose_h", "pose_ink",
                "sil_x0", "sil_x1", "sil_y0", "sil_y1"]
        if csv_path:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = _csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in cols})

        def _col(name):
            v = np.array([float(r[name]) for r in rows
                          if isinstance(r.get(name), (int, float))
                          and not isinstance(r.get(name), bool)], dtype=np.float64)
            return v[np.isfinite(v)] if v.size else v

        def _stat(name):
            v = _col(name)
            if v.size == 0:
                return "n/a"
            return (f"med={np.median(v):.3f} p95={np.percentile(v, 95):.3f} "
                    f"max={v.max():.3f}")

        def _toggles(name):
            seq = [1 if r.get(name) else 0 for r in rows]
            return sum(1 for a, b in zip(seq, seq[1:]) if a != b), sum(seq)

        n = len(rows)

        def _p(line):
            # stdout is already mirrored into the report by _StdoutTee
            print(line)

        _p("[BetaSwap][DIAG] ===== per-frame summary =====")
        hr, hv = _col("hf_recon"), _col("hf_vit")
        if hr.size and hv.size:
            ratio = np.median(hr) / max(float(np.median(hv)), 1e-6)
            verdict = ("recon manufactures the shake -> raise driver_jitter_filter"
                       if ratio > 1.5 else
                       "recon is as clean as ViTPose -> the shake is real driver motion, "
                       "a filter would only add lag")
            print(f"[BetaSwap][DIAG] high-freq noise (5-frame quadratic residual, same "
                  f"joints): recon {_stat('hf_recon')} | ViTPose {_stat('hf_vit')}")
            print(f"[BetaSwap][DIAG] hf_recon / hf_vit = {ratio:.2f}  -> {verdict}")
        ch = _col("prompt_churn")
        if ch.size:
            npr = _col("n_prompts")
            print(f"[BetaSwap][DIAG] prompt set: {np.median(npr):.0f} joints/frame, "
                  f"churn (joints entering+leaving vs previous frame) "
                  f"{_stat('prompt_churn')}; frames with churn>0: "
                  f"{int((ch > 0).sum())}/{ch.size}")
        hraw = _col("hf_raw")
        if hraw.size:
            print(f"[BetaSwap][DIAG] recon noise before the zero-lag filter: {_stat('hf_raw')}")
        print(f"[BetaSwap][DIAG] driver-recon shake   dri_jerk px   {_stat('dri_jerk')}"
              f"  <- keypoint prompts change THIS; the offset EMA passes it through by design")
        print(f"[BetaSwap][DIAG] swap before smoothing raw_jerk px  {_stat('swap_raw_jerk')}")
        print(f"[BetaSwap][DIAG] swap after  smoothing out_jerk px  {_stat('swap_out_jerk')}")
        print(f"[BetaSwap][DIAG] driver motion / frame dri_step px  {_stat('dri_step')}")
        se = _col("s_eff")
        if se.size:
            print(f"[BetaSwap][DIAG] s_eff min={se.min():.4f} max={se.max():.4f} "
                  f"med={np.median(se):.4f}; frames below the 0.999 threshold="
                  f"{int((se < 0.999).sum())}/{se.size}")
        tw, cw = _toggles("warp")
        tc, cc = _toggles("clear")
        tsl, csl = _toggles("sil")
        print(f"[BetaSwap][DIAG] mask-warp on {cw}/{n} frames, switched {tw}x | "
              f"head-clearance on {cc}/{n}, switched {tc}x | silhouette-union on "
              f"{csl}/{n}, switched {tsl}x  (a switch = hard mask geometry change "
              f"between two frames)")
        mf = _col("mask_frac")
        if mf.size:
            print(f"[BetaSwap][DIAG] mask pixels that are neither 0 nor 1: "
                  f"{_stat('mask_frac')} of the frame  (a region mask must be "
                  f"binary; anything above 0 here is partial-strength conditioning)")
        hr2 = _col("headroom_px")
        if hr2.size:
            print(f"[BetaSwap][DIAG] headroom above the swap head (mask top to head "
                  f"top): {_stat('headroom_px')} px = {_stat('headroom_blk')} blocks "
                  f"of 32px  (this is the room anything worn on the head has to be "
                  f"drawn in; <=0 means no room at all)")
        print(f"[BetaSwap][DIAG] 32px mask blocks flipping per frame: before hold "
              f"{_stat('blocks_flip_pre')} | after hold {_stat('blocks_flip_post')}")
        print(f"[BetaSwap][DIAG] face sx {_stat('face_sx')} | hand_l {_stat('hand_l')} "
              f"| offset {_stat('off_med')}")
        _p("[BetaSwap][DIAG+] ===== size / volume chain =====")
        vf = _col("vol_factor")
        if vf.size:
            _p(f"[BetaSwap][DIAG+] build factor (mesh girth ratio, no beta read): "
               f"{_stat('vol_factor')} -> stick thickness only")
        ep, ed = _col("ear_pre"), _col("ear_draw")
        if ep.size and ed.size:
            _r = np.median(ed) / max(np.median(ep), 1e-6)
            _p(f"[BetaSwap][DIAG+] ear span px: reconstructed {_stat('ear_pre')} -> drawn "
               f"{_stat('ear_draw')} = x{_r:.3f}  "
               f"({'OK, head untouched' if abs(_r - 1.0) < 0.02 else 'WARNING: head is being rescaled somewhere'})")
        if _ZONE_LAST.get("zones"):
            _p("[BetaSwap][DIAG+] photo zone ratios ref/driver: "
               + " ".join(f"{k}={v:.3f}" for k, v in _ZONE_LAST["zones"].items()))
        mo = _col("mask_over_sil")
        if mo.size:
            _p(f"[BetaSwap][DIAG+] mask / body silhouette: {_stat('mask_over_sil')} "
               f"(budget {_MASK_BUDGET:.2f}; runs that invented furniture or "
               f"backdrop all measured above 1.9)")
        hn, hd = _col("hw_need_px"), _col("hw_deficit_px")
        if hn.size:
            _p(f"[BetaSwap][DIAG+] headwear room: needed {_stat('hw_need_px')} px | "
               f"opened {_stat('hw_up_px')} px | still short {_stat('hw_deficit_px')} px "
               f"(negative = fits)")
        hm = _col("head_w_mesh")
        if hm.size and ed.size:
            _p(f"[BetaSwap][DIAG+] mesh head silhouette width {_stat('head_w_mesh')} px vs "
               f"drawn ear span med={np.median(ed):.1f} px -> ear/mesh_head = "
               f"{np.median(ed) / max(np.median(hm), 1e-6):.3f}  (a fixed anatomical ratio: "
               f"if it moves between runs, the pose head is drifting away from the body "
               f"mesh it is supposed to belong to)")
        sp_, sd = _col("sh_pre"), _col("sh_draw")
        if sp_.size and sd.size:
            _p(f"[BetaSwap][DIAG+] shoulder span px: {_stat('sh_pre')} -> drawn "
               f"{_stat('sh_draw')} = x{np.median(sd) / max(np.median(sp_), 1e-6):.3f}")
        pw = _col("pose_w")
        if pw.size:
            _p(f"[BetaSwap][DIAG+] drawn pose bbox: width {_stat('pose_w')} px | height "
               f"{_stat('pose_h')} px | top y {_stat('pose_y0')} | ink {_stat('pose_ink')} px")
            _p("[BetaSwap][DIAG+]   that bbox is the entire size signal Wan gets. "
               "Compare it run to run: it is the gabarity number.")
        hp = _col("headroom_pose_px")
        if hp.size:
            _p(f"[BetaSwap][DIAG+] headroom above the drawn head (mask top to "
               f"crown): {_stat('headroom_pose_px')} px.")
        sx0, sx1 = _col("sil_x0"), _col("sil_x1")
        if sx0.size and sx1.size:
            _p(f"[BetaSwap][DIAG+] swap silhouette bbox: x med [{np.median(sx0):.0f},"
               f"{np.median(sx1):.0f}] width med={np.median(sx1 - sx0):.0f} px | y top "
               f"{_stat('sil_y0')}")
        if csv_path:
            _p(f"[BetaSwap][DIAG] CSV written: {csv_path}")
    except Exception as e:
        import traceback as _tb2
        print(f"[BetaSwap][DIAG] report failed: {type(e).__name__}: {e}\n"
              f"{_tb2.format_exc()}")


# Savitzky-Golay smoothing weights, 5 samples, quadratic, evaluated at the
# CENTRE sample. Any constant-acceleration motion passes exactly, so the
# residual against the raw sample is pure high-frequency noise - the number
# that separates "the driver really moved" from "the recon shook".
_SG5 = np.array([-3.0, 12.0, 17.0, 12.0, -3.0]) / 35.0


def _hf_noise(window, idxs=None):
    """window: list of 5 (N,2) arrays, oldest first (NaN allowed). Returns the
    median per-point deviation of the centre sample from the quadratic fit."""
    if window is None or len(window) < 5:
        return float("nan")
    try:
        arr = np.stack([np.asarray(w, dtype=np.float64) for w in window], axis=0)
    except Exception:
        return float("nan")
    if idxs is not None:
        arr = arr[:, np.asarray(idxs, dtype=np.int64), :]
    fit = np.tensordot(_SG5, arr, axes=(0, 0))
    dev = np.linalg.norm(arr[2] - fit, axis=-1)
    dev = dev[np.isfinite(dev)]
    return float(np.median(dev)) if dev.size else float("nan")


def _posedata_body_pixels(driver_pose_data, frame_idx, src_H, src_W):
    """Driver POSEDATA body joints (ViTPose) remapped to MHR indices, in
    pixels, as a (70,2) array with NaN where a joint is missing. Same
    coordinate space as the recon's pred_keypoints_2d, so the two can be
    compared point for point."""
    meta = _get_pose_meta(driver_pose_data, frame_idx)
    out = np.full((70, 2), np.nan, dtype=np.float64)
    if meta is None:
        return out
    aa = getattr(meta, "kps_body", None)
    if aa is None:
        return out
    aa = np.asarray(aa, dtype=np.float64)
    if aa.shape != (20, 2):
        return out
    if aa.size and np.nanmax(np.abs(aa)) <= 1.5:
        aa = aa * np.array([[src_W, src_H]], dtype=np.float64)
    for mhr_idx, aa_idx in _AA_TO_MHR.items():
        out[mhr_idx] = aa[aa_idx]
    return out


# window -> measured white-noise attenuation (std_out / std_in)
_ZL_ATTEN = {5: 0.702, 7: 0.584, 9: 0.512, 11: 0.462, 13: 0.425}


def _zl_window_for(attenuation_needed):
    # smallest window that reaches the required attenuation
    for w in sorted(_ZL_ATTEN):
        if _ZL_ATTEN[w] <= attenuation_needed:
            return w
    return max(_ZL_ATTEN)


def _zero_lag_poly_smooth(seq, strength=1.0, poly=2, window=None):
    """seq: (T, N, 2). Centered polynomial (Savitzky-Golay) fit per point.

    Any motion that is polynomial of degree <= poly across the window passes
    through EXACTLY and with ZERO LAG - that includes constant velocity and
    constant acceleration, i.e. everything real human motion looks like over
    ~0.3 s. What does not survive is the per-frame residual: exactly the
    quantity hf_recon measures, and exactly what the keypoint-prompt refine
    head adds. A causal filter cannot do this - it would trade the shake for
    lag, and lag against an unfiltered driver mask is its own artifact.

    strength picks the window (wider = more attenuation, longer stretch of
    motion assumed smooth): 0.2->5, 0.4->7, 0.6->9, 0.8->11, 1.0->13 frames.
    White-noise attenuation, measured: 0.70 / 0.58 / 0.50 / 0.45 / 0.42.
    Edges use a one-sided fit over whatever frames exist."""
    seq = np.asarray(seq, dtype=np.float64)
    s = float(min(max(strength, 0.0), 1.0))
    T = seq.shape[0]
    if T < poly + 2 or (window is None and s <= 0.0):
        return seq.copy()
    w = int(window) if window is not None else int(round(3.0 + 10.0 * s))
    if w % 2 == 0:
        w += 1
    w = int(min(max(w, 3), max(3, T if T % 2 else T - 1)))
    half = w // 2
    flat = seq.reshape(T, -1)
    out = flat.copy()
    for t in range(T):
        lo, hi = max(0, t - half), min(T, t + half + 1)
        n = hi - lo
        if n < poly + 1:
            continue
        tt = np.arange(lo, hi, dtype=np.float64) - float(t)
        A = np.vander(tt, poly + 1)
        coef, _res, _rk, _sv = np.linalg.lstsq(A, flat[lo:hi], rcond=None)
        out[t] = coef[-1]              # polynomial value at tt = 0
    return out.reshape(seq.shape)


def _scale_face_to_swap(face_xy, kp2d_swap, kp2d_dri, strength, ema_state,
                        ema_alpha=0.5):
    """Scale driver dlib68(+eyeballs) about the nose tip (dlib idx 30) by the
    swap/driver head-size ratio: sx from eye+ear spans, sy from nose-to-eye-mid
    height. Expression (relative landmark offsets) is preserved, only the
    proportions follow the reference head. Axis-aligned anisotropy; at the
    clamped +-35% range the tilt error is negligible. EMA over frames to
    suppress per-frame recon jitter."""
    try:
        def _span(kp, a, b):
            return float(np.linalg.norm(
                np.asarray(kp[a], dtype=np.float64)
                - np.asarray(kp[b], dtype=np.float64)))
        sx_c = []
        for a, b in ((1, 2), (3, 4)):  # eyes, ears
            d = _span(kp2d_dri, a, b)
            s = _span(kp2d_swap, a, b)
            if d > 2.0 and s > 0.5:
                sx_c.append(s / d)
        eye_mid_d = (np.asarray(kp2d_dri[1], dtype=np.float64)
                     + np.asarray(kp2d_dri[2], dtype=np.float64)) / 2.0
        eye_mid_s = (np.asarray(kp2d_swap[1], dtype=np.float64)
                     + np.asarray(kp2d_swap[2], dtype=np.float64)) / 2.0
        hd = float(np.linalg.norm(np.asarray(kp2d_dri[0], dtype=np.float64) - eye_mid_d))
        hs = float(np.linalg.norm(np.asarray(kp2d_swap[0], dtype=np.float64) - eye_mid_s))
        sx = float(np.median(sx_c)) if sx_c else 1.0
        sy = (hs / hd) if (hd > 2.0 and hs > 0.5) else sx
        sx = min(max(sx, 0.75), 1.35)
        sy = min(max(sy, 0.75), 1.35)
        sx = 1.0 + float(strength) * (sx - 1.0)
        sy = 1.0 + float(strength) * (sy - 1.0)
        p_sx = ema_state.get("sx")
        if p_sx is not None:
            a = float(min(max(ema_alpha, 0.0), 0.95))
            sx = a * p_sx + (1.0 - a) * sx
            sy = a * ema_state.get("sy", sy) + (1.0 - a) * sy
        ema_state["sx"] = sx
        ema_state["sy"] = sy
        pivot = np.asarray(face_xy[30], dtype=np.float64)
        out = np.asarray(face_xy, dtype=np.float64).copy()
        out[:, 0] = pivot[0] + (out[:, 0] - pivot[0]) * sx
        out[:, 1] = pivot[1] + (out[:, 1] - pivot[1]) * sy
        return out.astype(np.float32)
    except Exception as e:
        print(f"[BetaSwap] face_shape: skipped ({type(e).__name__}: {e})")
        return face_xy


def _resolve_mask_for_run(mask_tensor):
    if mask_tensor is None:
        return None, None
    mask_np = comfy_mask_to_numpy(mask_tensor)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]
    rows = np.any(mask_np > 0.5, axis=1)
    cols = np.any(mask_np > 0.5, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bbox = np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)
    else:
        bbox = None
    return mask_np, bbox

def _build_estimator_factory(loaded):
    sam_3d_model = loaded["model"]
    model_cfg = loaded["model_cfg"]

    def factory():
        from ..sam_3d_body import SAM3DBodyEstimator
        return SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
    return factory

def _run_sam3d_with_optional_prompts(
    estimator,
    image_tensor_or_bgr,
    bbox_xyxy_array=None,
    mask_np=None,
    bbox_threshold=0.5,
    inference_type="full",
    keypoint_prompt_pixels=None,
):
    if isinstance(image_tensor_or_bgr, np.ndarray):
        img_bgr = image_tensor_or_bgr


    else:
        img_bgr = comfy_image_to_numpy(image_tensor_or_bgr)
    height, width = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    if bbox_xyxy_array is not None:
        boxes = bbox_xyxy_array.reshape(-1, 4)
    else:
        boxes = np.array([0, 0, width, height]).reshape(1, 4)

    masks_score = None
    if mask_np is not None:
        masks = (mask_np.reshape(-1, height, width, 1) > 0.5).astype(np.uint8)
        masks_score = np.ones(len(masks), dtype=np.float32)
    else:
        masks = None
        masks_score = None

    from ..sam_3d_body.data.utils.prepare_batch import prepare_batch
    from ..sam_3d_body.utils import recursive_to

    batch = prepare_batch(img_rgb, estimator.transform, boxes, masks, masks_score)
    batch = recursive_to(batch, estimator.device)
    estimator.model._initialize_batch(batch)

    _ = batch["cam_int"].clone()

    outputs = estimator.model.run_inference(
        img_rgb,
        batch,
        inference_type=inference_type,
        transform_hand=estimator.transform_hand,
        thresh_wrist_angle=estimator.thresh_wrist_angle,
    )
    if inference_type == "full":
        pose_output, batch_lhand, batch_rhand, _, _ = outputs
    else:
        pose_output = outputs

    if (
        keypoint_prompt_pixels is not None
        and len(keypoint_prompt_pixels) > 0
        and inference_type in ("full", "body")
        and hasattr(estimator.model, "run_keypoint_prompt")
        and hasattr(estimator.model, "prompt_encoder")
    ):
        prompt_tensor = _build_prompt_tensor_from_pixels(


            estimator, batch, keypoint_prompt_pixels
        )
        if prompt_tensor is not None and prompt_tensor.shape[1] > 0:
            try:
                refine_input = {
                    "image_embeddings": pose_output["image_embeddings"],
                    "condition_info": pose_output["condition_info"],
                    "mhr": pose_output["mhr"],
                }
                refined, _ = estimator.model.run_keypoint_prompt(
                    batch, refine_input, prompt_tensor
                )
                pose_output["mhr"] = refined["mhr"]
            except Exception as e:
                print(f"[BetaSwap] WARN keypoint-prompt refine failed, "
                      f"fallback to base recon: {type(e).__name__}: {e}")

    out = pose_output["mhr"]
    out = recursive_to(out, "cpu")
    out = recursive_to(out, "numpy")

    out_dict = {
        "bbox": batch["bbox"][0, 0].cpu().numpy(),
        "focal_length": out["focal_length"][0],
        "pred_keypoints_3d": out["pred_keypoints_3d"][0],
        "pred_keypoints_2d": out["pred_keypoints_2d"][0],
        "pred_vertices": out["pred_vertices"][0],
        "pred_cam_t": out["pred_cam_t"][0],
        "pred_pose_raw": out["pred_pose_raw"][0],
        "global_rot": out["global_rot"][0],
        "body_pose_params": out["body_pose"][0],
        "hand_pose_params": out["hand"][0],
        "scale_params": out["scale"][0],
        "shape_params": out["shape"][0],
        "expr_params": out["face"][0],
        "mask": masks[0] if masks is not None else None,
        "pred_joint_coords": out["pred_joint_coords"][0],
        "pred_global_rots": out["joint_global_rots"][0],
    }
    return out_dict, img_bgr

def _build_prompt_tensor_from_pixels(estimator, batch, keypoint_prompt_pixels):
    if not isinstance(keypoint_prompt_pixels, np.ndarray):
        keypoint_prompt_pixels = np.asarray(keypoint_prompt_pixels, dtype=np.float32)
    if keypoint_prompt_pixels.ndim != 2 or keypoint_prompt_pixels.shape[1] != 3:
        return None


    valid = keypoint_prompt_pixels[:, 2] >= -1
    pts = keypoint_prompt_pixels[valid]
    if len(pts) == 0:
        return None

    device = estimator.device
    img = batch["img"]
    affine_trans = batch["affine_trans"][0, 0].cpu().numpy()
    input_size = batch["img_size"][0, 0].cpu().numpy()
    Wc, Hc = float(input_size[0]), float(input_size[1])

    xy = pts[:, :2].astype(np.float32)
    xy_h = np.concatenate([xy, np.ones((len(xy), 1), dtype=np.float32)], axis=1)
    xy_crop = xy_h @ affine_trans.T
    xy_norm = np.stack([xy_crop[:, 0] / Wc, xy_crop[:, 1] / Hc], axis=-1)

    out_of_bounds = (
        (xy_norm[:, 0] < 0.0) | (xy_norm[:, 0] > 1.0)
        | (xy_norm[:, 1] < 0.0) | (xy_norm[:, 1] > 1.0)
    )
    xy_norm = np.clip(xy_norm, 0.0, 1.0)
    labels = pts[:, 2].copy().astype(np.float32)
    labels[out_of_bounds] = -1.0

    prompt = np.concatenate([xy_norm, labels[:, None]], axis=-1).astype(np.float32)
    prompt_tensor = torch.from_numpy(prompt).unsqueeze(0).to(
        device=device, dtype=img.dtype
    )
    return prompt_tensor

def _beta_swap_forward(sam_3d_model, driver_out, ref_out, device, dtype=torch.float32,
                       shape_strength=1.0, scale_strength=1.0, amplify_reference=1.0,
                       ref_body_out=None):
    def _to_t(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return x.to(device=device, dtype=dtype).unsqueeze(0) if x.ndim == 1 else x.to(device=device, dtype=dtype)

    global_rot       = _to_t(driver_out["global_rot"])
    body_pose_full   = _to_t(driver_out["body_pose_params"])
    body_pose_params = body_pose_full[:, :130]
    hand_pose_params = _to_t(driver_out["hand_pose_params"])
    expr_params      = _to_t(driver_out.get("expr_params"))

    driver_shape = _to_t(driver_out["shape_params"])
    driver_scale = _to_t(driver_out["scale_params"])


    # Body FORM (shape betas + bone scale) source:
    # If a dedicated body reference recon is supplied (ref_body_out), the torso/limb
    # form is taken from it - this is the correct source when the main reference is a
    # head/face crop that cannot express body proportions. The main reference (ref_out)
    # still governs identity/face elsewhere in the pipeline. When ref_body_out is None
    # (solo-reference projects), form falls back to the main reference exactly as before.
    _form_out = ref_body_out if ref_body_out is not None else ref_out
    ref_shape    = _to_t(_form_out["shape_params"])
    ref_scale    = _to_t(_form_out["scale_params"])

    shape_blend = driver_shape + shape_strength * (ref_shape - driver_shape)
    scale_blend = driver_scale + scale_strength * (ref_scale - driver_scale)

    shape_params = driver_shape + amplify_reference * (shape_blend - driver_shape)
    scale_params = driver_scale + amplify_reference * (scale_blend - driver_scale)

    global_trans = torch.zeros_like(global_rot)
    mhr_head = sam_3d_model.head_pose

    with torch.no_grad():
        output = mhr_head.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot,
            body_pose_params=body_pose_params,
            hand_pose_params=hand_pose_params,
            scale_params=scale_params,
            shape_params=shape_params,
            expr_params=expr_params,
            return_keypoints=True,
            do_pcblend=True,
        )

    if isinstance(output, (tuple, list)):
        verts, j3d_308 = output[0], output[1]
    else:
        raise RuntimeError("mhr_forward returned unexpected single tensor")

    j3d_308_np = j3d_308.detach().cpu().numpy()[0]
    j3d_70 = j3d_308_np[:70].copy()
    # Axis flip: public mhr_forward does not apply it; forward()/run_inference
    # flip verts+j3d+jcoords identically (verified in heads.py / model.py).
    j3d_70[..., [1, 2]] *= -1
    verts_np = None
    if verts is not None:
        verts_np = verts.detach().cpu().numpy()[0].copy()
        verts_np[..., [1, 2]] *= -1
    return j3d_70, verts_np

def _build_aapose_meta(kp2d_mhr70, H, W, default_conf=1.0):
    body_xy = kp2d_mhr70[MHR_TO_WAN20]
    body_p  = np.full(20, default_conf, dtype=np.float32)

    lhand_xy = kp2d_mhr70[MHR_TO_OPENPOSE_LHAND]
    rhand_xy = kp2d_mhr70[MHR_TO_OPENPOSE_RHAND]
    lhand_p  = np.full(21, default_conf, dtype=np.float32)
    rhand_p  = np.full(21, default_conf, dtype=np.float32)

    meta = AAPoseMeta()


    meta.width = int(W)
    meta.height = int(H)
    meta.kps_body = body_xy.astype(np.float32)
    meta.kps_body_p = body_p
    meta.kps_lhand = lhand_xy.astype(np.float32)
    meta.kps_lhand_p = lhand_p
    meta.kps_rhand = rhand_xy.astype(np.float32)
    meta.kps_rhand_p = rhand_p
    meta.kps_face = np.zeros((70, 2), dtype=np.float32)
    meta.kps_face_p = np.zeros(70, dtype=np.float32)
    return meta

def _get_pose_meta(driver_pose_data, frame_idx):
    if driver_pose_data is None or not isinstance(driver_pose_data, dict):
        return None
    metas = driver_pose_data.get("pose_metas")
    if metas is None or len(metas) == 0 or frame_idx >= len(metas):
        return None
    return metas[frame_idx]

def _ensure_pixel_xy(xy, conf, n_expected, src_H, src_W):
    if xy is None:
        return None, None
    xy = np.asarray(xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] != n_expected or xy.shape[1] < 2:
        return None, None
    xy = xy[:, :2].copy()
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float32)
        if conf.shape[0] != n_expected:
            conf = np.full(n_expected, 1.0, dtype=np.float32)
    else:
        conf = np.full(n_expected, 1.0, dtype=np.float32)
    if xy.size > 0 and xy.max() <= 1.5:
        xy[:, 0] *= src_W
        xy[:, 1] *= src_H
    return xy, conf

def _extract_driver_hands_from_pose_data(driver_pose_data, frame_idx, src_H, src_W):
    meta = _get_pose_meta(driver_pose_data, frame_idx)
    if meta is None:
        return None, None, None, None
    lhand_xy, lhand_conf = _ensure_pixel_xy(
        getattr(meta, "kps_lhand", None), getattr(meta, "kps_lhand_p", None),


        21, src_H, src_W,
    )
    rhand_xy, rhand_conf = _ensure_pixel_xy(
        getattr(meta, "kps_rhand", None), getattr(meta, "kps_rhand_p", None),
        21, src_H, src_W,
    )
    return lhand_xy, lhand_conf, rhand_xy, rhand_conf

_FACE_LAYOUT_LOGGED = False

def _extract_driver_face_from_pose_data(driver_pose_data, frame_idx, src_H, src_W):
    """Return (face_xy_70, face_conf_70) in absolute pixels.

    Runtime kijai layout: kps_face has 69 elements where index 0 is R_heel
    (COCO[22]) and indices [1..68] are dlib68 face. Offline kp2ds layout
    delivers 70 (dlib68 + 2 COCO eye centers). Direct dlib68 (68) also handled.
    """
    global _FACE_LAYOUT_LOGGED

    meta = _get_pose_meta(driver_pose_data, frame_idx)
    if meta is None:
        return None, None
    raw_xy = getattr(meta, "kps_face", None)
    raw_p  = getattr(meta, "kps_face_p", None)
    if raw_xy is None:
        return None, None
    raw_xy = np.asarray(raw_xy, dtype=np.float32)
    if raw_xy.ndim != 2 or raw_xy.shape[1] < 2:
        return None, None
    n_src = raw_xy.shape[0]

    if not _FACE_LAYOUT_LOGGED:
        layout = "Runtime split (heel+dlib68)" if n_src == 69 else (
                 "Offline kp2ds (dlib68+COCO eyes)" if n_src == 70 else
                 "Direct dlib68" if n_src == 68 else f"Unknown {n_src}")
        print(f"[BetaSwap] kijai POSEDATA face layout: shape=({n_src},{raw_xy.shape[1]}) -> {layout}")
        _FACE_LAYOUT_LOGGED = True

    raw_xy = raw_xy[:, :2].copy()
    if raw_p is None:
        raw_p = np.full(n_src, 1.0, dtype=np.float32)
    else:
        raw_p = np.asarray(raw_p, dtype=np.float32)
        if raw_p.shape[0] != n_src:
            raw_p = np.full(n_src, 1.0, dtype=np.float32)


    if raw_xy.size > 0 and raw_xy.max() <= 1.5:
        raw_xy[:, 0] *= src_W
        raw_xy[:, 1] *= src_H

    xy_70 = np.zeros((70, 2), dtype=np.float32)
    p_70  = np.zeros(70, dtype=np.float32)

    if n_src == 69:
        xy_70[:68] = raw_xy[1:69]
        p_70[:68]  = raw_p[1:69]
        synth_eyeballs = True
    elif n_src == 70:
        xy_70[:70] = raw_xy[:70]
        p_70[:70]  = raw_p[:70]
        synth_eyeballs = False
    elif n_src == 68:
        xy_70[:68] = raw_xy[:68]
        p_70[:68]  = raw_p[:68]
        synth_eyeballs = True
    else:
        return None, None

    if synth_eyeballs:
        if (p_70[36:42] > 0.3).all():
            xy_70[69] = xy_70[36:42].mean(axis=0)
            p_70[69]  = float(p_70[36:42].mean())
        if (p_70[42:48] > 0.3).all():
            xy_70[68] = xy_70[42:48].mean(axis=0)
            p_70[68]  = float(p_70[42:48].mean())

    return xy_70, p_70

def _build_keypoint_prompt_pixels_from_pose_data(driver_pose_data, frame_idx,
                                                 src_H, src_W, conf_threshold=0.3,
                                                 state=None):
    """Build (M, 3) array of (x_pixel, y_pixel, mhr_label) from kijai POSEDATA
    AA-body (20 pts). Returns None if no usable points.
    """
    meta = _get_pose_meta(driver_pose_data, frame_idx)
    if meta is None:
        return None

    aa_body = getattr(meta, "kps_body", None)
    aa_body_p = getattr(meta, "kps_body_p", None)
    if aa_body is None:
        return None


    aa_body = np.asarray(aa_body, dtype=np.float32)
    if aa_body.shape != (20, 2):
        return None
    if aa_body_p is None:
        aa_body_p = np.ones(20, dtype=np.float32)
    else:
        aa_body_p = np.asarray(aa_body_p, dtype=np.float32)

    # Which joints are prompted is decided per frame by a hard confidence
    # threshold, so a joint hovering near it enters and leaves the constraint
    # set between frames - and the refine head then solves a DIFFERENT problem
    # each frame. The release band keeps an already-prompted joint in until its
    # confidence drops well below the threshold, so the set stops flickering.
    prev = state.get("in", set()) if isinstance(state, dict) else set()
    low = float(conf_threshold) * _KP_PROMPT_RELEASE
    cur = set()
    prompts = []
    for mhr_idx, aa_idx in _AA_TO_MHR.items():
        p = float(aa_body_p[aa_idx])
        keep = (p >= conf_threshold) or (mhr_idx in prev and p >= low)
        if not keep:
            continue
        cur.add(mhr_idx)
        x, y = aa_body[aa_idx]
        prompts.append([float(x), float(y), int(mhr_idx)])

    if isinstance(state, dict):
        state["churn"] = len(cur ^ prev) if state.get("in") is not None else 0
        state["in"] = cur

    if not prompts:
        return None
    return np.asarray(prompts, dtype=np.float32)

class SAM3DBodyBetaSwapPoseRender:

    @classmethod
    def INPUT_TYPES(cls):


        return {
            "required": {
                "sam3d_model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model config from LoadSAM3DBodyModel"
                }),
                "driver_images": ("IMAGE", {
                    "tooltip": "Driver video frames [N, H, W, 3]"
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Single reference image [1, H, W, 3]"
                }),
                "target_width": ("INT", {"default": 720, "min": 64, "max": 4096, "step": 8}),
                "target_height": ("INT", {"default": 1280, "min": 64, "max": 4096, "step": 8}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1,
                    "tooltip": "-1 = auto from canvas size, 0 = disable body, >0 = explicit width"}),
                "hand_stick_width": ("INT", {"default": 4, "min": 0, "max": 20, "step": 1,
                    "tooltip": "0 = disable hand drawing"}),
                "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "head_anchor_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Post-projection 2D shift of head joints (nose, eyes, ears) so swap "
                               "nose lands on driver nose pixel. 0.0 = off, 1.0 = full anchor."}),
                "shape_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Body shape (blendshapes) interpolation weight. 0.0 = driver, "
                               "1.0 = full reference. Safe: 0.0-1.3."}),
                "scale_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Bone scale (limb length) interpolation weight. Safe: 0.0-1.3."}),
                "amplify_reference": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Post-blend multiplier on identity delta. 1.0 = no-op. Safe: 0.0-1.3."}),
                "draw_face": ("BOOLEAN", {"default": False,
                    "tooltip": "Render face landmarks (jaw, brows, eyes, nose, mouth, eyeballs) "
                               "directly from driver POSEDATA dlib68."}),
                "use_keypoint_prompts": ("BOOLEAN", {"default": False,
                    "tooltip": "Refine MHR recon by feeding driver POSEDATA body keypoints into "
                               "SAM3DBody.run_keypoint_prompt() after the dummy-prompt forward."}),
                "kp_prompt_conf_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Confidence floor for driver POSEDATA keypoints when used as "
                               "prompts. Only used when use_keypoint_prompts=True."}),
                "force_height_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01,
                    "tooltip": "Real height change in the frame. Figure is shifted in DEPTH (Z) so "
                               "projected size becomes baseline*scale uniformly across body, then "
                               "pelvis pixel is anchored to driver pelvis pixel. Driver mask is "
                               "warped uniformly around pelvis pixel by the same scale so Wan "
                               "regenerates background in the freed region (s<1) or extends the "
                               "subject region (s>1). 1.0 = no-op."}),
            },
            "optional": {
                "reference_mask": ("MASK", {"tooltip": "Optional mask for reference"}),


                "reference_target_mask": ("MASK", {
                    "tooltip": "Per-reference mask of target person. When provided, its bbox "
                               "drives ref MHR recon. Takes precedence over reference_mask."}),
                "driver_masks": ("MASK", {"tooltip": "Optional per-frame masks [N, H, W]"}),
                "driver_pose_data": ("POSEDATA", {
                    "tooltip": "kijai PoseAndFaceDetection.pose_data output. Required for "
                               "draw_face=True or use_keypoint_prompts=True. Driver hands are "
                               "always taken from this input."}),
                "volume_from_reference": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "0.0 = AUTOMATIC (recommended): the build difference measured between "
                               "the reference and the driver (MHR body-mass beta) is applied as it "
                               "is - a heavier reference draws thicker bones and a proportionally "
                               "larger head, a lighter one draws thinner. Nothing to tune: whatever "
                               "the reference actually is, is what comes out. Above 0 exaggerates "
                               "(or damps) the measured difference by that factor."}),
                # RESTORED: these two sockets were present in the earlier build and
                # vanished from INPUT_TYPES in the current one, while run() still takes
                # them as kwargs. Without the declaration ComfyUI never passes them, so
                # a connected body reference is silently ignored and body FORM falls
                # back to the main reference. Sockets are not widgets: adding them here
                # does not move a single positional value in widgets_values.
                "reference_body_image": ("IMAGE", {
                    "tooltip": "OPTIONAL full-body or waist-up reference, used ONLY to measure body "
                               "proportions/volume when the main reference_image is a head/face crop. "
                               "Identity and face are still taken from the main reference_image; this "
                               "input only supplies body form (shape betas + bone scale). If empty, "
                               "form is measured from the main reference_image (legacy single-ref "
                               "behavior). Single-image projects can ignore this input entirely."}),
                "reference_body_mask": ("MASK", {
                    "tooltip": "Optional mask for reference_body_image (bbox drives its body recon). "
                               "Also used as the clothed-silhouette source for "
                               "clothing_volume_strength when connected (best precision)."}),
                "face_shape_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Scale driver dlib68 face landmarks about the nose tip by the "
                               "swap/driver head-size ratio (eye+ear spans for width, nose-to-eye "
                               "height for height), so face GEOMETRY follows the reference head "
                               "proportions while EXPRESSION stays 100% driver. "
                               "0.0 = raw driver face (old behavior)."}),
                "mask_from_swap": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Union the projected SWAP body silhouette (MHR mesh with the "
                               "reference shape/scale) into transformed_driver_mask, then snap to "
                               "the 32px block grid. Gives Wan room to draw a wider/bigger build "
                               "than the driver instead of being capped by the driver "
                               "segmentation. 0.0 = passthrough (old behavior)."}),
                "clothing_volume_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Measure how much WIDER the clothed reference silhouette is than "
                               "its minimal MHR body (per zone: torso / thigh / shin) and re-add "
                               "that clothing/muscle volume: widens the swap-silhouette mask rows "
                               "by the full ratio and the shoulder/hip spread by half of it. "
                               "Silhouette source: reference_body_mask if connected, else "
                               "automatic GrabCut seeded by the projected mesh. "
                               "0.0 = off (old behavior)."}),
                "auto_height_from_reference": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Automatically transfer the reference HEIGHT: measures the "
                               "swap/driver skeletal height ratio (pose-invariant bone-chain "
                               "sums, EMA over frames) and feeds it into the force_height "
                               "machinery (depth shift + pelvis pin + mask warp). Manual "
                               "force_height_scale multiplies on top as a trim. Note: scaling "
                               "happens about the pelvis, so foot/ground contact shifts with "
                               "strong height changes. 0.0 = off (old behavior)."}),
                "temporal_smooth": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Kills the per-frame shake that appears with "
                               "use_keypoint_prompts WITHOUT giving up their accuracy. EMA is "
                               "applied to the swap-MINUS-driver OFFSET field, not to the "
                               "points: driver motion (stable POSEDATA-refined recon) passes "
                               "through 1:1 with no lag, only the SAM3D per-frame recon jitter "
                               "is averaged out. Also EMAs the lstsq intrinsics (static camera, "
                               "noisy per-frame fit), the face size ratio, and fades the 32px "
                               "mask-block union over frames so blocks stop popping. "
                               "0.6 ~ halves the jitter, 0.8 ~ cuts it 3x (cost: 1-2 frames of "
                               "offset lag on fast pose changes). 0.0 = bit-exact old behavior."}),
                "diagnostics": ("BOOLEAN", {"default": True,
                    "tooltip": "Measure-only instrumentation: changes NO output pixel. Writes a "
                               "per-frame CSV (jerk of the driver recon vs the swap skeleton "
                               "before/after smoothing, s_eff, warp/clearance flags, 32px mask "
                               "block flips, face and hand scale factors) next to ComfyUI's "
                               "output folder, prints a summary, and dumps a reference-mesh "
                               "report + overlay PNG showing why clothing_volume accepts or "
                               "rejects the measurement. Turn off for production runs."}),
                "driver_jitter_filter": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0.0 = AUTOMATIC (recommended). The node measures, on this clip, the "
                               "per-frame noise of the driver recon and of the POSEDATA detections "
                               "it is fitting to, and smooths the recon only as far as needed to "
                               "stop it being noisier than its own input - no further. Smoothing is "
                               "a centred polynomial fit, so motion of any speed passes with zero "
                               "lag and only the per-frame residual is removed; when the recon is "
                               "already clean (keypoint prompts off) nothing is applied at all. "
                               "Any value above 0 overrides the measurement with a fixed window "
                               "(0.2->5 frames ... 1.0->13)."}),
                "kp_prompt_hysteresis": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05,
                    "tooltip": "RETIRED - ignored. The prompt-set release band is automatic now "
                               "(a joint leaves the set only when its confidence drops below half "
                               "the threshold). The DIAG column prompt_churn reports what is left."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("pose_images", "debug_overlay", "transformed_driver_mask")
    FUNCTION = "run"
    CATEGORY = "SAM3DBody/WanAnimate"
    DESCRIPTION = (
        "beta-swap: per-frame MHR recon on driver, run mhr_forward with ref identity, "
        "project to driver camera, render Wan-Animate pose_images. Driver hands are taken "
        "from kijai POSEDATA. Optional dlib68 face overlay and keypoint-prompt-conditioned "
        "recon refinement."
    )

    def run(self, *args, **kwargs):
        """Thin wrapper: everything printed inside lands in the report file too."""
        _log = _DiagLog(None)
        self._active_log = _log
        try:
            with _StdoutTee(_log):
                return self._run_impl(*args, **kwargs)
        finally:
            _log.flush()
            self._active_log = None

    def _run_impl(self, sam3d_model, driver_images, reference_image,
            target_width, target_height,
            body_stick_width, hand_stick_width,
            bbox_threshold,
            head_anchor_strength=1.0,
            shape_strength=1.0, scale_strength=1.0, amplify_reference=1.0,
            draw_face=False,
            use_keypoint_prompts=False, kp_prompt_conf_threshold=0.3,
            force_height_scale=1.0,
            reference_mask=None, reference_target_mask=None,
            driver_masks=None,
            driver_pose_data=None,
            volume_from_reference=0.0,
            reference_body_image=None,
            reference_body_mask=None,
            face_shape_strength=1.0, mask_from_swap=1.0,
            clothing_volume_strength=1.0,
            auto_height_from_reference=1.0,
            temporal_smooth=0.6,
            diagnostics=True,
            driver_jitter_filter=0.0,
            kp_prompt_hysteresis=0.0):  # kept for widget-slot compatibility; unused

        if not _KIJAI_FOUND:
            raise RuntimeError(
                f"ComfyUI-WanAnimatePreprocess not found. Error: {_KIJAI_ERROR}"
            )

        FACE_CONF_GATE = 0.5

        if driver_pose_data is None:
            raise RuntimeError(
                "driver_pose_data is required (driver hands always come from kijai "
                "POSEDATA). Connect PoseAndFaceDetection.pose_data."
            )


        if isinstance(driver_pose_data, dict):
            metas = driver_pose_data.get("pose_metas")
            if metas is not None and len(metas) != driver_images.shape[0]:
                print(f"[BetaSwap] WARN: driver_pose_data has {len(metas)} frames but "
                      f"driver_images has {driver_images.shape[0]}. Mismatched frames "
                      f"will fall back to no-prompt / no-driver-overlay.")

        N = driver_images.shape[0]
        draw_hand = hand_stick_width != 0

        loaded = _load_sam3d_model(sam3d_model)
        sam_3d_model = loaded["model"]
        device = torch.device(loaded["device"])
        factory = _build_estimator_factory(loaded)

        print(f"[BetaSwap] shape={shape_strength:.2f} scale={scale_strength:.2f} "
              f"amplify={amplify_reference:.2f} | head_anchor={head_anchor_strength:.2f} | "
              f"draw_face={draw_face} | kp_prompts={use_keypoint_prompts} | "
              f"force_height={force_height_scale:.2f} | "
              f"face_shape={face_shape_strength:.2f} | mask_from_swap={mask_from_swap:.2f} | "
              f"clothing_vol={clothing_volume_strength:.2f} | "
              f"auto_height={auto_height_from_reference:.2f} | "
              f"t_smooth={temporal_smooth:.2f} | "
              f"dri_filter={'auto' if driver_jitter_filter <= 0 else f'{driver_jitter_filter:.2f}'} | "
              f"kp_gate=auto")

        _ts = float(min(max(temporal_smooth, 0.0), 0.95))

        _ZONE_LAST.clear()
        _diag_rows = []
        _diag_csv = None   # kept as a name only; nothing is written to disk now
        _diag_png = None
        _diag_txt = None
        _dlog = getattr(self, "_active_log", None) or _DiagLog(None)
        if diagnostics:
            try:
                import time as _time
                try:
                    import folder_paths as _fp
                    _dd = _fp.get_output_directory()
                except Exception:
                    _dd = os.path.dirname(os.path.abspath(__file__))
                _stamp = _time.strftime("%Y%m%d_%H%M%S")
                _diag_txt = os.path.join(_dd, f"betaswap_report_{_stamp}.txt")
                _dlog.path = _diag_txt
                print(f"[BetaSwap][DIAG] instrumentation ON (no effect on outputs). "
                      f"Single report file -> {_diag_txt}")
                _dlog("BetaSwap report " + _stamp)
            except Exception as _e:
                print(f"[BetaSwap][DIAG] could not prepare the report path: {_e}")
                _diag_txt = None

        # Reference recon
        print(f"[BetaSwap] Running SAM3DBody on reference image...")
        ref_estimator = factory()
        ref_mask_input = reference_target_mask if reference_target_mask is not None else reference_mask
        _, ref_mask_bbox = _resolve_mask_for_run(ref_mask_input)
        ref_mask_np = None
        if ref_mask_input is not None:
            tmp_mask = comfy_mask_to_numpy(ref_mask_input)
            if tmp_mask.ndim == 3:
                tmp_mask = tmp_mask[0]
            ref_mask_np = tmp_mask

        ref_out, ref_bgr = _run_sam3d_with_optional_prompts(
            ref_estimator,
            reference_image,
            bbox_xyxy_array=ref_mask_bbox,
            mask_np=ref_mask_np,
            bbox_threshold=bbox_threshold,
            inference_type="full",
            keypoint_prompt_pixels=None,
        )
        if ref_out.get("shape_params") is None or ref_out.get("scale_params") is None:
            raise RuntimeError("Reference recon missing shape_params/scale_params")

        # --- Body reference recon (separate full-body / waist-up image) ---
        # When reference_body_image is provided, run a dedicated SAM3DBody recon on it.
        # Its shape betas + bone scale become the BODY FORM source in _beta_swap_forward
        # (correct when the main reference is a head/face crop). Identity/face still come
        # from the main reference. When absent, body form falls back to the main reference
        # (legacy solo-reference behavior). This recon runs once, before the frame loop.
        _body_out = None
        _body_bgr = None
        if reference_body_image is not None:
            try:
                _body_mask_input = reference_body_mask
                _, _body_bbox = _resolve_mask_for_run(_body_mask_input)
                _body_mask_np = None
                if _body_mask_input is not None:
                    _bm = comfy_mask_to_numpy(_body_mask_input)
                    if _bm.ndim == 3:
                        _bm = _bm[0]
                    _body_mask_np = _bm
                _body_estimator = factory()
                _body_out, _body_bgr = _run_sam3d_with_optional_prompts(
                    _body_estimator,
                    reference_body_image,
                    bbox_xyxy_array=_body_bbox,
                    mask_np=_body_mask_np,
                    bbox_threshold=bbox_threshold,
                    inference_type="full",
                    keypoint_prompt_pixels=None,
                )
                if _body_out.get("shape_params") is None or _body_out.get("scale_params") is None:
                    print("[BetaSwap] reference_body_image recon missing shape/scale; "
                          "ignoring it, body form will come from main reference.")
                    _body_out = None
                else:
                    print("[BetaSwap] reference_body_image recon OK -> body FORM (shape+scale) "
                          "taken from body image; identity/face from main reference.")
            except Exception as _e:
                print(f"[BetaSwap] reference_body_image recon failed ({_e}); "
                      f"body form will come from main reference.")
                _body_out = None

        # --- volume_from_reference: build difference, measured on the meshes ---
        # No beta index is read any more. The reference shape vector and the
        # driver shape vector are posed identically and the two meshes are
        # measured; the ratio of their height-normalised girth is the factor.
        # Computed once on the first frame, then held for the clip.
        _vol_strength = 1.0 if volume_from_reference <= 0.0 else float(volume_from_reference)
        _vol_auto = volume_from_reference <= 0.0
        _build_src = _body_out if _body_out is not None else ref_out
        _ref_shape = None
        _vol_factor_run = None
        if _vol_strength > 0.0:
            try:
                _ref_shape = _vec(_build_src["shape_params"])
                _src_name = "reference_body_image" if _body_out is not None else "main reference"
                print(f"[BetaSwap] volume_from_reference="
                      f"{'auto' if _vol_auto else f'{_vol_strength:.2f}'} "
                      f"(build measured from {_src_name} mesh vs driver mesh) -> stick thickness only.")
            except Exception as _e:
                print(f"[BetaSwap] volume_from_reference: no shape vector, disabling. ({_e})")
                _ref_shape = None

        # --- headwear reserve: how much mask room whatever is on the head needs.
        # Filled in from the reference further down, held for the whole clip.
        _hw_rise = 0.0
        _hw_half = 0.0
        _ref_prof = None
        _zone_ratio = None
        _stature_adj = None
        _mesh_h_ratio = None

        # --- DIAG+: the raw parameter vectors, verbatim, plus the one experiment
        # that settles what index 1 actually means. Everything bulky goes to the
        # text report, only the verdict reaches the console.
        if diagnostics:
            try:
                _dlog("")
                _dlog("===== raw shape/scale parameters =====")
                for _nm, _o in (("main reference", ref_out),
                                ("reference_body_image", _body_out)):
                    if _o is None:
                        _dlog(f"{_nm}: absent")
                        continue
                    _sv = _vec(_o.get("shape_params"))
                    _cv = _vec(_o.get("scale_params"))
                    _dlog(f"{_nm}: shape_params[{0 if _sv is None else _sv.shape[0]}] = "
                          + ("None" if _sv is None else
                             " ".join(f"{i}:{v:+.4f}" for i, v in enumerate(_sv))))
                    _dlog(f"{_nm}: scale_params[{0 if _cv is None else _cv.shape[0]}] = "
                          + ("None" if _cv is None else
                             " ".join(f"{i}:{v:+.4f}" for i, v in enumerate(_cv))))
                _probe_src = _body_out if _body_out is not None else ref_out
                _probe_name = ("reference_body_image" if _body_out is not None
                               else "main reference")
                _diag_beta_probe(sam_3d_model, _probe_src, device,
                                 _probe_name, _dlog)
            except Exception as _e:
                import traceback as _tbp
                print(f"[BetaSwap][DIAG+] parameter dump / beta probe failed: "
                      f"{type(_e).__name__}: {_e}")
                _dlog("parameter dump / beta probe failed:")
                _dlog(_tbp.format_exc())

        # --- clothing_volume: measure clothed-vs-minimal-body width ratios (once) ---
        # SAM3D/MHR regresses a parametric BODY (skinned template + shape/scale
        # blendshapes): clothing geometry is not part of the mesh, so a bulky
        # outfit does not survive into the betas. Re-measure the lost volume
        # straight from the reference photo: clothed silhouette (user mask or
        # mesh-seeded GrabCut) vs the projected minimal mesh, per height zone.
        # --- read both reference photographs once: bare mesh silhouette,
        # clothed silhouette, 2D skeleton. Everything below is measured off
        # these, so no measurement depends on a shape coefficient.
        _packs = []
        for _pl, _po, _pi in (("main reference", ref_out, ref_bgr),
                              ("reference_body_image", _body_out, _body_bgr)):
            if _po is None:
                continue
            _pk = _ref_photo_pack(_po, _pi, sam_3d_model, device, _pl,
                                  _dlog if diagnostics else None)
            if _pk is not None:
                _packs.append(_pk)

        # --- headwear / long hair. The MHR mesh is bald and earless, so
        # whatever sits above and beside its crown IS the hat, the ears, the
        # hair. Previous build only looked at the main reference - a tight head
        # crop where the accessory is outside the frame - and so reserved
        # nothing while the body reference was showing 0.92 head-heights of
        # rabbit ears. Scan every reference, keep the largest, skip any whose
        # silhouette runs off the top edge (cropped, so unmeasurable).
        for _pk in _packs:
            if _pk["touches_top"]:
                if diagnostics:
                    _dlog(f"headwear: {_pk['label']} is cropped at the top edge, "
                          f"skipped as unmeasurable")
                continue
            try:
                _r, _h = _measure_headwear(_pk["img"], _pk["mesh_sil"], _pk["kp"],
                                           _dlog if diagnostics else None,
                                           _pk["label"])
                if _r > _hw_rise:
                    _hw_rise, _hw_half = _r, _h
            except Exception as _e:
                print(f"[BetaSwap] headwear on {_pk['label']} skipped ({_e})")
        if _hw_rise > 0.02:
            print(f"[BetaSwap] headwear/hair: reference carries {_hw_rise:.2f} "
                  f"ear-spans of height above the crown and {_hw_half:.2f} "
                  f"sideways -> mask opened by that much every frame")
        else:
            print("[BetaSwap] headwear/hair: nothing measurable above any "
                  "reference head, no extra mask room reserved")

        # --- build profile of the reference photo (widest silhouette wins:
        # a knee crop and a full body describe the same person, but the one
        # with more of the body visible describes more zones).
        _ref_prof = None
        for _pk in _packs:
            _pr = _silhouette_profile(_pk["sil"], _pk["kp"], _pk["label"],
                                      _dlog if diagnostics else None)
            _pm = _silhouette_profile(_pk["mesh_sil"], _pk["kp"],
                                      _pk["label"] + " bare mesh",
                                      _dlog if diagnostics else None)
            _pr = _validate_profile(_pr, _pm, _pk["label"],
                                    _dlog if diagnostics else None)
            if _pr is not None and (_ref_prof is None
                                    or len(_pr) > len(_ref_prof)):
                _ref_prof = _pr

        _cloth = None
        if clothing_volume_strength > 0.0:
            try:
                _cl_src = _body_out if _body_out is not None else ref_out
                _cl_img = _body_bgr if _body_out is not None else ref_bgr
                # Regenerate the ref body through the SAME machinery as the
                # in-loop silhouette (mhr_forward at strengths=0 reproduces the
                # recon's own body) and project it with the same lstsq-fitted
                # intrinsics. This is the path whose on-screen correctness is
                # directly visible in the debug overlay.
                _cl_verts = None
                _cl_int = None
                _cl_j3d = None
                try:
                    _cl_j3d, _cl_verts = _beta_swap_forward(
                        sam_3d_model, _cl_src, _cl_src, device,
                        shape_strength=0.0, scale_strength=0.0,
                        amplify_reference=1.0, ref_body_out=None)
                    _cl_int = _fit_intrinsics(
                        _cl_src["pred_keypoints_3d"], _cl_src["pred_cam_t"],
                        _cl_src["pred_keypoints_2d"])
                except Exception as _e:
                    print(f"[BetaSwap] clothing_volume: mesh regen failed "
                          f"({type(_e).__name__}: {_e}); falling back.")
                _cl_mask = None
                if _body_out is not None and reference_body_mask is not None:
                    _cm = comfy_mask_to_numpy(reference_body_mask)
                    _cl_mask = _cm[0] if _cm.ndim == 3 else _cm
                elif _body_out is None:
                    _rm = reference_target_mask if reference_target_mask is not None else reference_mask
                    if _rm is not None:
                        _cm = comfy_mask_to_numpy(_rm)
                        _cl_mask = _cm[0] if _cm.ndim == 3 else _cm
                if diagnostics:
                    _diag_reference_mesh(_cl_src, _cl_j3d, _cl_verts, _cl_int,
                                         _cl_img, save_path=None)
                    _diag_ref_framing(_cl_src, _cl_img, _cl_verts, _cl_j3d,
                                      _cl_int,
                                      "body-form source", _dlog)
                    if _body_out is not None:
                        try:
                            _mj, _mv = _beta_swap_forward(
                                sam_3d_model, ref_out, ref_out, device,
                                shape_strength=0.0, scale_strength=0.0,
                                amplify_reference=1.0, ref_body_out=None)
                            _mi = _fit_intrinsics(ref_out["pred_keypoints_3d"],
                                                  ref_out["pred_cam_t"],
                                                  ref_out["pred_keypoints_2d"])
                            _diag_ref_framing(ref_out, ref_bgr, _mv, _mj, _mi,
                                              "main reference", _dlog)
                        except Exception as _e:
                            _dlog(f"main-reference framing failed: {_e}")
                _cloth = _measure_clothing_ratios(_cl_src, _cl_img, _cl_mask,
                                                  mesh_verts=_cl_verts,
                                                  cam_int=_cl_int,
                                                  mesh_j3d=_cl_j3d)
                if _cloth is not None:
                    print(f"[BetaSwap] clothing_volume: silhouette/mesh width ratios "
                          f"torso={_cloth['torso']:.2f} thigh={_cloth['thigh']:.2f} "
                          f"shin={_cloth['shin']:.2f} (raw={_cloth['raw']}, "
                          f"source: {_cloth['source']}, "
                          f"root-frame align={_cloth.get('align')}, "
                          f"work={_cloth.get('work')}, "
                          f"strength={clothing_volume_strength:.2f})")
                    _sk_f = min(1.0 + 0.35 * float(clothing_volume_strength)
                                * (max(_cloth["torso"], 1.0) - 1.0), 1.25)
                    _st_f = min(1.0 + float(clothing_volume_strength)
                                * max(_cloth["torso"] - 1.0, 0.0), 1.5)
                    print(f"[BetaSwap] clothing_volume -> mask rows x{_cloth['torso']:.2f}, "
                          f"shoulder/hip spread x{_sk_f:.3f} (damped 0.35, cap 1.25), "
                          f"stick thickness x{_st_f:.2f} (cap 1.5)")
                else:
                    print("[BetaSwap] clothing_volume: measurement unavailable "
                          "(no vertices / degenerate silhouette) -> disabled this run.")
            except Exception as _e:
                print(f"[BetaSwap] clothing_volume: failed "
                      f"({type(_e).__name__}: {_e}) -> disabled.")
                _cloth = None

        _face_ema = {}
        # --- temporal smoothing state ---
        # _off_ema : EMA of the swap-minus-driver 2D offset field (70,2)
        # _int_ema : EMA of the lstsq intrinsics [fx, fy, cx, cy]
        # _hold    : decaying trace of the mask union (32px block anti-flicker)
        _off_ema = None
        _int_ema = None
        _hold = None
        _smooth_frames = 0
        # diagnostics history (previous / previous-previous frame arrays)
        _dg_dri1 = _dg_dri2 = None
        _dg_raw1 = _dg_raw2 = None
        _dg_out1 = _dg_out2 = None
        _dg_mask_prev = None
        _dg_maskpre_prev = None
        _dg_win_dri = []
        _dg_win_vit = []
        _sil_frames = 0
        _ah_ratio = None
        _warp_frames = 0
        _s_last = float(force_height_scale)
        _chain_state = None
        _chain_prev = None
        _anchor_face_frames = 0
        _clear_frames = 0
        _hw_frames = 0
        _trim_frames = 0

        pose_images_out = []
        debug_overlay_out = []
        transformed_masks_out = []


        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(N)
        except Exception:
            pbar = None

        hands_substituted = 0
        face_rendered_count = 0
        prompts_used_total = 0
        prompts_frames_with = 0

        driver_estimator = factory()

        _prompt_state = {}
        _prompt_stats = {}

        def _prep_mask(i, frame):
            mask_for_frame = None
            mask_bbox = None
            if driver_masks is not None:
                if driver_masks.ndim == 3:
                    mask_t = driver_masks[i:i+1]
                else:
                    mask_t = driver_masks
                tmp_mask = comfy_mask_to_numpy(mask_t)
                if tmp_mask.ndim == 3:
                    tmp_mask = tmp_mask[0]
                mask_for_frame = tmp_mask
                rows = np.any(mask_for_frame > 0.5, axis=1)
                cols = np.any(mask_for_frame > 0.5, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    mask_bbox = np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)
            return mask_for_frame, mask_bbox

        def _recon_one(i, frame):
            nonlocal prompts_frames_with, prompts_used_total
            mask_for_frame, mask_bbox = _prep_mask(i, frame)
            prompt_pixels = None
            if use_keypoint_prompts:
                src_H = int(frame.shape[1])
                src_W = int(frame.shape[2])
                prompt_pixels = _build_keypoint_prompt_pixels_from_pose_data(
                    driver_pose_data, i, src_H, src_W,
                    conf_threshold=kp_prompt_conf_threshold,
                    state=_prompt_state,
                )
                if prompt_pixels is not None:
                    prompts_frames_with += 1
                    prompts_used_total += len(prompt_pixels)
            _prompt_stats[i] = (0 if prompt_pixels is None else int(len(prompt_pixels)),
                                int(_prompt_state.get("churn", 0)))
            dri_out, dri_bgr = _run_sam3d_with_optional_prompts(
                driver_estimator,
                frame,
                bbox_xyxy_array=mask_bbox,
                mask_np=mask_for_frame,
                bbox_threshold=bbox_threshold,
                inference_type="full",
                keypoint_prompt_pixels=prompt_pixels,
            )
            return dri_out, dri_bgr, mask_for_frame, mask_bbox

        # --- optional pre-pass: zero-lag smoothing needs the whole sequence,
        # so the recon runs for every frame first and the rendering loop then
        # reads the (filtered) results. Only the recon dicts are kept; the BGR
        # frame is regenerated in the loop, which is free.
        _recon_cache = []
        _hf_raw_pre = {}
        if True:
            _recon_cache = []
            for i in range(N):
                _d, _b, _m, _bb = _recon_one(i, driver_images[i:i+1])
                _recon_cache.append(_d)
                if pbar is not None:
                    pbar.update_absolute(int(i * 0.5))
            try:
                _seq = np.stack([np.asarray(d["pred_keypoints_2d"], dtype=np.float64)
                                 for d in _recon_cache])
                _vit = np.stack([_posedata_body_pixels(
                    driver_pose_data, _k, int(driver_images.shape[1]),
                    int(driver_images.shape[2])) for _k in range(N)])
                _idxs = [k for k in _AA_TO_MHR.keys()
                         if np.all(np.isfinite(_vit[:, k, :]))]

                def _hf_series(a, idxs=None):
                    return [(_hf_noise([a[k] for k in range(t - 4, t + 1)], idxs)
                             if t >= 4 else float("nan")) for t in range(len(a))]

                _hr = _hf_series(_seq, _idxs)
                _hv = _hf_series(_vit, _idxs)
                _hr_a = np.array([v for v in _hr if np.isfinite(v)])
                _hv_a = np.array([v for v in _hv if np.isfinite(v)])

                # --- automatic decision -------------------------------------
                # The fit must not be noisier than the detections it is fitting
                # to. If it is, smooth it exactly down to that level and stop
                # there; if it is already cleaner, leave it alone. Nothing to
                # dial: the clip decides.
                _w = None
                if driver_jitter_filter > 0.0:
                    _w = int(round(3.0 + 10.0 * float(driver_jitter_filter)))
                    _w += (1 if _w % 2 == 0 else 0)
                    _why = f"manual override ({driver_jitter_filter:.2f})"
                elif _hr_a.size and _hv_a.size and np.median(_hv_a) > 1e-6:
                    _ratio = float(np.median(_hr_a) / np.median(_hv_a))
                    if _ratio > 1.0:
                        _w = _zl_window_for(1.0 / _ratio)
                        _why = (f"recon noise {np.median(_hr_a):.2f} px vs POSEDATA "
                                f"{np.median(_hv_a):.2f} px = x{_ratio:.2f} too noisy")
                    else:
                        print(f"[BetaSwap] driver_jitter_filter=auto: recon noise "
                              f"{np.median(_hr_a):.2f} px is at or below its own input "
                              f"({np.median(_hv_a):.2f} px) -> no smoothing applied")
                if _w is not None:
                    _seq_f = _zero_lag_poly_smooth(_seq, window=_w)
                    _gain = _ZL_ATTEN.get(_w, 0.5)
                    _left = float(np.median(_hr_a) * _gain) if _hr_a.size else 0.0
                    _capped = ("; widest window reached, some excess remains"
                               if (_hv_a.size and _left > float(np.median(_hv_a)) * 1.02)
                               else "")
                    print(f"[BetaSwap] driver_jitter_filter=auto: {_why} -> "
                          f"{_w}-frame centred fit, white-noise attenuation x{_gain:.2f} "
                          f"(-> ~{_left:.2f} px){_capped}. Motion of any speed passes "
                          f"with zero lag.")
                    for _i2, _d in enumerate(_recon_cache):
                        _dtype = np.asarray(_d["pred_keypoints_2d"]).dtype
                        _d["pred_keypoints_2d"] = _seq_f[_i2].astype(_dtype)
                        _hf_raw_pre[_i2] = _hr[_i2]
            except Exception as _e:
                import traceback as _tb3
                print(f"[BetaSwap] driver_jitter_filter failed, using raw recon: "
                      f"{type(_e).__name__}: {_e}\n{_tb3.format_exc()}")

        for i in range(N):
            frame = driver_images[i:i+1]
            _dg = {"frame": i} if diagnostics else None

            if _recon_cache:
                dri_out = _recon_cache[i]
                dri_bgr = comfy_image_to_numpy(frame)
                mask_for_frame, mask_bbox = _prep_mask(i, frame)
            else:
                dri_out, dri_bgr, mask_for_frame, mask_bbox = _recon_one(i, frame)

            if _dg is not None:
                _np_, _ch_ = _prompt_stats.get(i, (0, 0))
                _dg["n_prompts"] = _np_
                _dg["prompt_churn"] = _ch_
                if i in _hf_raw_pre and np.isfinite(_hf_raw_pre[i]):
                    _dg["hf_raw"] = round(float(_hf_raw_pre[i]), 4)

            kp3d_dri = dri_out["pred_keypoints_3d"]
            kp2d_dri = dri_out["pred_keypoints_2d"]
            cam_t    = dri_out["pred_cam_t"]

            # --- noise accounting: the same 5-frame quadratic-residual metric
            # on the recon joints and on the driver POSEDATA (ViTPose) joints.
            # Same points, same frames: if hf_recon >> hf_vit the shake is
            # manufactured by the reconstruction, not by the driver.
            if _dg is not None:
                try:
                    _vit_now = _posedata_body_pixels(driver_pose_data, i,
                                                     int(frame.shape[1]), int(frame.shape[2]))
                    _idxs = [k for k in _AA_TO_MHR.keys()
                             if np.all(np.isfinite(_vit_now[k]))]
                    _dg_win_dri.append(np.asarray(kp2d_dri, dtype=np.float64).copy())
                    _dg_win_vit.append(_vit_now)
                    if len(_dg_win_dri) > 5:
                        _dg_win_dri.pop(0)
                        _dg_win_vit.pop(0)
                    if len(_dg_win_dri) == 5 and _idxs:
                        _hr = _hf_noise(_dg_win_dri, _idxs)
                        _hv = _hf_noise(_dg_win_vit, _idxs)
                        if np.isfinite(_hr):
                            _dg["hf_recon"] = round(float(_hr), 4)
                        if np.isfinite(_hv):
                            _dg["hf_vit"] = round(float(_hv), 4)
                except Exception:
                    pass


            # --- volume_from_reference: per-frame factor (drives BOTH thickness & head balance) ---
            # factor>1 => reference is fuller than driver (thicken skeleton, enlarge head to match);
            # factor<1 => reference is leaner (thin skeleton, shrink head to match). factor==1 => no-op.
            eff_body_stick_width = body_stick_width
            _vol_factor = 1.0
            if _ref_shape is not None or _ref_prof is not None:
                if _vol_factor_run is None:
                    _r = None
                    _how = ""
                    # 1st choice: both silhouettes, photo vs photo.
                    try:
                        if _ref_prof is not None and mask_for_frame is not None:
                            _dp = _silhouette_profile(
                                (np.clip(mask_for_frame, 0.0, 1.0) > 0.5).astype(np.uint8),
                                kp2d_dri, "driver frame 0",
                                _dlog if diagnostics else None)
                            # the driver side gets the same plausibility band;
                            # a broken driver mask would poison every ratio
                            _dp = _validate_profile(_dp, None, "driver frame 0",
                                                    _dlog if diagnostics else None)
                            _g, _z = _photo_build_ratio(
                                _ref_prof, _dp, _dlog if diagnostics else None)
                            if _g is not None:
                                _r, _how = _g, "photo silhouettes (reference vs driver)"
                                _zone_ratio = _z
                                _ZONE_LAST["zones"] = _z
                                # A garment cannot make a leg wider than the
                                # torso slack allows, and if the photographs say
                                # a zone is slimmer than the driver there is
                                # nothing there to widen. The cape measured
                                # thigh 1.56 and shin 3.18 against a torso slack
                                # of 1.03 - a skin-tight suit has no slack, so
                                # all of that was cape. The mask ballooned at the
                                # hips and the model filled the balloon with an
                                # object that is not on the reference.
                                if _cloth is not None:
                                    _tcap = 1.0 + 1.25 * max(
                                        float(_cloth["torso"]) - 1.0, 0.0)
                                    _before = {k: float(_cloth[k])
                                               for k in ("thigh", "shin")
                                               if k in _cloth}
                                    for _zn in ("thigh", "shin"):
                                        if _zn not in _cloth:
                                            continue
                                        _pz = _z.get(_zn, _z.get("thigh"))
                                        _v = min(float(_cloth[_zn]), _tcap)
                                        if _pz is not None and _pz < 0.95:
                                            _v = min(_v, 1.05)
                                        _cloth[_zn] = float(max(_v, 0.9))
                                    if _before:
                                        print("[BetaSwap] leg zones re-capped by "
                                              "torso slack and the photo ratio: "
                                              + ", ".join(
                                                  f"{k} {v:.2f}->{_cloth[k]:.2f}"
                                                  for k, v in _before.items())
                                              + f" (torso slack cap {_tcap:.2f})")
                    except Exception as _e:
                        print(f"[BetaSwap] photo build failed ({_e})")
                    # 2nd choice: the meshes. Known to under-read a clothed
                    # reference, so it is only used when there is no driver mask.
                    if _r is None and _ref_shape is not None:
                        try:
                            _r, _det = _build_ratio(
                                sam_3d_model, dri_out, device, _ref_shape,
                                _vec(dri_out["shape_params"]),
                                _dlog if diagnostics else None)
                            _how = "meshes (no driver mask to measure)"
                        except Exception as _e:
                            print(f"[BetaSwap] mesh build ratio failed ({_e})")
                    if _r is None:
                        _r, _how = 1.0, "nothing measurable"
                    # Stature. The skeletal chain reads two characters whose
                    # meshes differ 7.5% in height as only 4.2% apart, because a
                    # joint chain misses the flesh at both ends. Measure both
                    # meshes in the same pose and nudge s_eff toward what they
                    # say - hard-bounded to +/-4%, because height feeds the warp,
                    # the mask and the head clearance, and this is the one place
                    # where a wrong number damages everything downstream.
                    if _ref_shape is not None:
                        try:
                            _vr, _jr = _mhr_verts_for_shape(
                                sam_3d_model, dri_out, device, _ref_shape)
                            _vd, _jd = _mhr_verts_for_shape(
                                sam_3d_model, dri_out, device,
                                _vec(dri_out["shape_params"]))
                            _hr = _mesh_shape_metrics(_vr, _jr)["height"]
                            _hd = _mesh_shape_metrics(_vd, _jd)["height"]
                            _mesh_h_ratio = float(_hr / max(_hd, 1e-6))
                            print(f"[BetaSwap] stature: reference mesh {_hr:.3f}m vs "
                                  f"driver mesh {_hd:.3f}m = {_mesh_h_ratio:.4f}")
                        except Exception as _e:
                            print(f"[BetaSwap] stature measurement skipped ({_e})")
                    _vol_factor_run = float(min(max(
                        1.0 + _vol_strength * (float(_r) - 1.0), 0.65), 1.75))
                    print(f"[BetaSwap] build: reference vs driver = {float(_r):.4f} "
                          f"from {_how} x strength {_vol_strength:.2f} -> stick factor "
                          f"{_vol_factor_run:.4f} (head is NOT scaled by this)")
                _vol_factor = float(_vol_factor_run)
                if _dg is not None:
                    _dg["vol_factor"] = round(_vol_factor, 5)

            # Stick width: build factor and clothing slack are applied together and
            # rounded ONCE. Rounding after each of them separately loses up to a
            # whole pixel of stroke, and a pixel is 30-50% of the stroke here.
            if body_stick_width is not None and body_stick_width > 0:
                _base_w = float(body_stick_width)
            elif body_stick_width == 0:
                _base_w = 0.0
            else:
                # A 2px auto base quantises every build factor to the same
                # integer: 2 x 1.12 and 2 x 0.98 both round to 2, which is why
                # two characters measured 10% apart drew identically. A finer
                # base is the only way a measured difference reaches the canvas.
                _fh = int(frame.shape[1]); _fw = int(frame.shape[2])
                _base_w = float(max(int(round(min(_fh, _fw) / 180.0)), 2))
            # The build factor already compares the CLOTHED reference with the
            # CLOTHED driver, so the outfit is inside it. Multiplying by the
            # reference garment slack on top counted the same coat twice, and it
            # counted it against the difference we want: the girl in the puffy
            # jacket carried x1.09-1.16 of slack against x1.03-1.04 for a man in
            # a catsuit, cancelling her lower build ratio exactly. Garment slack
            # still drives the MASK, which is built from the bare mesh and does
            # need the room.
            if _base_w > 0.0:
                eff_body_stick_width = int(round(min(max(
                    _base_w * _vol_factor, 1.0), 20.0)))

            # (clothing slack is already folded into eff_body_stick_width above)

            j3d_cam_dri = kp3d_dri + cam_t
            xn = j3d_cam_dri[:, 0] / j3d_cam_dri[:, 2]
            yn = j3d_cam_dri[:, 1] / j3d_cam_dri[:, 2]
            Ax = np.stack([xn, np.ones_like(xn)], axis=-1)
            Ay = np.stack([yn, np.ones_like(yn)], axis=-1)
            fx, cx = np.linalg.lstsq(Ax, kp2d_dri[:, 0], rcond=None)[0]
            fy, cy = np.linalg.lstsq(Ay, kp2d_dri[:, 1], rcond=None)[0]
            # The camera does not move between frames, but this fit is re-solved
            # from a jittery recon every frame; EMA removes that noise without
            # touching anything the driver actually does. The anchor pin
            # unprojects and re-projects with the SAME matrix, so the anchor
            # still lands exactly on its driver pixel whatever the fit says.
            if _ts > 0.0:
                _iv = np.array([fx, fy, cx, cy], dtype=np.float64)
                if _int_ema is None or not np.all(np.isfinite(_int_ema)):
                    _int_ema = _iv
                else:
                    _int_ema = _ts * _int_ema + (1.0 - _ts) * _iv
                fx, fy, cx, cy = (float(v) for v in _int_ema)
            cam_int = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            j3d_swap, verts_swap = _beta_swap_forward(
                sam_3d_model, dri_out, ref_out, device,
                shape_strength=shape_strength,
                scale_strength=scale_strength,
                amplify_reference=amplify_reference,
                ref_body_out=_body_out,
            )

            # --- framing state: which driver landmarks are actually ON SCREEN.
            # In waist-up / close framing the pelvis, knees or ankles exist only
            # as recon hallucinations; the anchor and the height chain must not
            # rely on them. The face is the one landmark always in frame.
            _Hf = int(frame.shape[1])
            _Wf = int(frame.shape[2])

            def _dri_on_screen(idx):
                return (0.0 <= kp2d_dri[idx, 0] < _Wf) and (0.0 <= kp2d_dri[idx, 1] < _Hf)

            _pel_in = _dri_on_screen(_LHIP_IDX) and _dri_on_screen(_RHIP_IDX)
            if _pel_in and _dri_on_screen(13) and _dri_on_screen(14):
                _st_now = "full"
            elif _pel_in:
                _st_now = "torso"
            else:
                _st_now = "head"
            if _chain_state is None:
                _chain_state = _st_now
            elif _st_now != _chain_state and _chain_prev == _st_now:
                _chain_state = _st_now  # 2-frame confirm against popping
            _chain_prev = _st_now
            _face_anchor = (_chain_state == "head")
            if _face_anchor:
                _anchor_face_frames += 1

            # --- auto_height_from_reference: effective height scale for this
            # frame. Ratio of pose-invariant skeletal heights (swap vs driver)
            # over the OBSERVED chain, EMA-smoothed; manual force_height_scale
            # multiplies on top.
            s_eff = float(force_height_scale)
            if auto_height_from_reference > 0.0:
                try:
                    _r_now = (_chain_height(j3d_swap, _chain_state)
                              / max(_chain_height(kp3d_dri, _chain_state), 1e-6))
                    if _dg is not None:
                        _dg["ratio_raw"] = round(float(_r_now), 5)
                    if _stature_adj is None:
                        if _mesh_h_ratio is None:
                            _stature_adj = 1.0
                        else:
                            _stature_adj = float(min(max(
                                _mesh_h_ratio / max(float(_r_now), 1e-6),
                                0.96), 1.04))
                            print(f"[BetaSwap] stature: meshes say {_mesh_h_ratio:.4f}, "
                                  f"skeletal chain says {float(_r_now):.4f} -> height "
                                  f"nudged x{_stature_adj:.4f} (hard bound +/-4%)")
                    _ah_ratio = (_r_now if _ah_ratio is None
                                 else 0.7 * _ah_ratio + 0.3 * _r_now)
                    s_eff = float(min(max(
                        force_height_scale
                        * (1.0 + float(auto_height_from_reference) * (_ah_ratio - 1.0))
                        * (_stature_adj if _stature_adj else 1.0),
                        0.5), 1.5))
                except Exception as _e:
                    s_eff = float(force_height_scale)
            _s_last = s_eff
            if _dg is not None:
                _dg["s_eff"] = round(float(s_eff), 5)
                _dg["chain"] = _chain_state
                _dg["anchor"] = "face" if _face_anchor else "pelvis"
                _dg["stick_w"] = int(eff_body_stick_width) if eff_body_stick_width is not None else -1

            cam_t_swap = cam_t.astype(np.float64).copy()

            # Root anchor (always on): pin one OBSERVED driver landmark. Pelvis
            # when it is on screen (keeps ground behavior; scale pivots at the
            # pelvis); otherwise the NOSE - in waist-up content the face is the
            # only landmark guaranteed in frame, and a hallucinated off-frame
            # pelvis is not a reliable anchor. The same point is the pivot for
            # height scaling and for the mask warp, so in face mode the face
            # stays fixed and the body grows/shrinks downward from it.
            if _face_anchor:
                anchor_swap_3d = j3d_swap[0].astype(np.float64)
                _anchor_px = kp2d_dri[0].astype(np.float64)
            else:
                anchor_swap_3d = ((j3d_swap[_LHIP_IDX] + j3d_swap[_RHIP_IDX]) / 2).astype(np.float64)
                _anchor_px = ((kp2d_dri[_LHIP_IDX] + kp2d_dri[_RHIP_IDX]) / 2).astype(np.float64)
            anchor_cam_old = anchor_swap_3d + cam_t_swap
            Z_old = float(anchor_cam_old[2])
            fx_, fy_ = float(cam_int[0, 0]), float(cam_int[1, 1])
            cx_, cy_ = float(cam_int[0, 2]), float(cam_int[1, 2])
            if Z_old > 1e-3 and abs(fx_) > 1e-6 and abs(fy_) > 1e-6:
                Z_new = Z_old / s_eff
                X_new = (_anchor_px[0] - cx_) / fx_ * Z_new
                Y_new = (_anchor_px[1] - cy_) / fy_ * Z_new
                anchor_cam_new = np.array([X_new, Y_new, Z_new], dtype=np.float64)
                cam_t_swap = cam_t_swap + (anchor_cam_new - anchor_cam_old)

            j3d_cam_swap = j3d_swap + cam_t_swap
            kp2d_swap = _perspective_project(j3d_cam_swap, cam_int)

            # Head 2D anchor: nose+eyes+ears -> driver nose pixel
            if head_anchor_strength > 0.0:
                head_idxs_2d = np.array([0, 1, 2, 3, 4], dtype=np.int64)
                nose_dri_2d = kp2d_dri[0]
                if abs(s_eff - 1.0) > 1e-3 and not _face_anchor:
                    # Pelvis-pivot scaling moves every projected offset about
                    # the pelvis pixel by s_eff; anchoring the head to the RAW
                    # driver nose would undo the height gain and stretch the
                    # neck. Anchor to the scale-equivalent nose instead. In
                    # face mode the nose IS the pivot (invariant), so the raw
                    # driver nose is already the correct target.
                    _pel_px = (kp2d_dri[_LHIP_IDX] + kp2d_dri[_RHIP_IDX]) / 2.0
                    nose_dri_2d = _pel_px + (nose_dri_2d - _pel_px) * s_eff
                nose_swap_2d = kp2d_swap[0]
                delta_2d = (nose_dri_2d - nose_swap_2d) * float(head_anchor_strength)
                kp2d_swap = kp2d_swap.copy()
                kp2d_swap[head_idxs_2d] = kp2d_swap[head_idxs_2d] + delta_2d

            # --- volume_from_reference: HEAD/BODY BALANCE ---
            # When the body was thickened/thinned by _vol_factor, the head must scale
            # by the same factor or it looks disproportionate (tiny head on a fat body,
            # or huge head on a lean body). We scale the head joints (nose/eyes/ears)
            # about the NECK anchor in 2D, so head size tracks body mass while staying
            # attached at the neck. Uses the projected neck if available, else the
            # midpoint of the shoulders, else the nose (no-op pivot).
            # The face landmarks below must follow the head that was MEASURED on
            # the reference, not the head after the body-mass heuristic below
            # nudged it. Keep a copy of the five head points as reconstructed.
            _head_measured = np.asarray(kp2d_swap[:5], dtype=np.float64).copy()
            if _dg is not None:
                try:
                    _dg["ear_pre"] = round(float(np.linalg.norm(
                        _head_measured[3] - _head_measured[4])), 3)
                    _dg["sh_pre"] = round(float(np.linalg.norm(
                        np.asarray(kp2d_swap[5], dtype=np.float64)
                        - np.asarray(kp2d_swap[6], dtype=np.float64))), 3)
                except Exception:
                    pass
            # REMOVED: the head used to be scaled by the body-mass factor about the
            # nose. It was double counting - the swap head already comes from the
            # reference shape vector, whose head components are reconstructed from
            # the reference face - and measured x1.23 on one character and x0.87 on
            # another, i.e. the anatomical ear/head ratio swung 36%. The head is now
            # left exactly as reconstructed. ear_pre == ear_draw is the check.

            if _dg is not None:
                try:
                    _dg["ear_post"] = round(float(np.linalg.norm(
                        np.asarray(kp2d_swap[3], dtype=np.float64)
                        - np.asarray(kp2d_swap[4], dtype=np.float64))), 3)
                except Exception:
                    pass

            # --- clothing_volume: widen shoulder/hip spread (damped x0.35,
            # cap 1.25); whole arm/leg chains translate rigidly so limb shape is
            # preserved. The MASK gets the full measured ratio (the outfit needs
            # room to be drawn), but the SKELETON only a fraction: silhouette
            # width is mostly garment/soft tissue, not bone spread - pushing
            # joints by the full outfit ratio turns every reference into a
            # wardrobe (observed failure). 0.25/1.2 left heavy builds visibly
            # short of the reference, so the damper is raised - still far below
            # the full ratio, and widen-only (ratio<=1 is untouched).
            # v5: the spread is now driven by the PHOTO ratio (reference body vs
            # driver body, both in head widths) rather than by the reference
            # silhouette-vs-its-own-mesh slack. The old source produced x1.011 and
            # x1.032 for two characters that measure ~10% apart, i.e. it
            # transmitted nothing. Joint spread is the strongest size cue the
            # skeleton has, so it gets 0.7 of the measured difference; and unlike
            # the garment ratio it may now also go BELOW 1 - a slimmer character
            # must be drawn narrower, not merely "not wider".
            _fs = None
            _fs_src = None
            if _zone_ratio:
                _fz = [_zone_ratio[k] for k in ("chest", "waist")
                       if k in _zone_ratio]
                if _fz:
                    _fs_src = float(np.mean(_fz))
            if _fs_src is not None and clothing_volume_strength > 0.0:
                _fs = float(min(max(1.0 + 0.7 * float(clothing_volume_strength)
                                    * (_fs_src - 1.0), 0.80), 1.35))
            elif _cloth is not None and clothing_volume_strength > 0.0:
                _fs = min(1.0 + 0.35 * float(clothing_volume_strength)
                          * (max(_cloth["torso"], 1.0) - 1.0), 1.25)
            if _fs is not None:
                if _dg is not None:
                    _dg["spread"] = round(float(_fs), 4)
                if abs(_fs - 1.0) > 0.001:
                    kp2d_swap = kp2d_swap.astype(np.float64).copy()
                    _mid_sh = (kp2d_swap[5] + kp2d_swap[6]) / 2.0
                    for _j, _chain in ((5, _ARM_L_CHAIN), (6, _ARM_R_CHAIN)):
                        _d = (kp2d_swap[_j] - _mid_sh) * (_fs - 1.0)
                        kp2d_swap[_j] = kp2d_swap[_j] + _d
                        kp2d_swap[_chain] = kp2d_swap[_chain] + _d
                    _mid_hp = (kp2d_swap[_LHIP_IDX] + kp2d_swap[_RHIP_IDX]) / 2.0
                    for _j, _chain in ((_LHIP_IDX, _LEG_L_CHAIN), (_RHIP_IDX, _LEG_R_CHAIN)):
                        _d = (kp2d_swap[_j] - _mid_hp) * (_fs - 1.0)
                        kp2d_swap[_j] = kp2d_swap[_j] + _d
                        kp2d_swap[_chain] = kp2d_swap[_chain] + _d

            # --- temporal_smooth: EMA on the OFFSET FIELD, not on the points.
            # What the pose_images must follow 1:1 is the driver motion; what
            # shakes is the per-frame MHR/beta recon (worse with keypoint
            # prompts, because the refine step re-fits the body every frame).
            # Those two live in one array only as a sum: kp2d_swap =
            # kp2d_dri + (identity/height/volume offset). Smoothing kp2d_swap
            # itself would lag the motion; smoothing only the offset lets the
            # driver through untouched and averages exactly the recon noise.
            # The offset is a slowly-varying quantity by construction (same
            # identity all clip), so a strong EMA costs almost nothing.
            _dg_raw_now = (np.asarray(kp2d_swap, dtype=np.float64).copy()
                           if _dg is not None else None)
            if _ts > 0.0:
                _off = (np.asarray(kp2d_swap, dtype=np.float64)
                        - np.asarray(kp2d_dri, dtype=np.float64))
                if (_off_ema is None or _off_ema.shape != _off.shape
                        or not np.all(np.isfinite(_off_ema))):
                    _off_ema = _off
                else:
                    _off_ema = _ts * _off_ema + (1.0 - _ts) * _off
                kp2d_swap = np.asarray(kp2d_dri, dtype=np.float64) + _off_ema
                _smooth_frames += 1

            if _dg is not None:
                try:
                    _d_now = np.asarray(kp2d_dri, dtype=np.float64)
                    _o_now = np.asarray(kp2d_swap, dtype=np.float64)
                    _dg["dri_step"] = round(float(np.median(np.linalg.norm(
                        _d_now - _dg_dri1, axis=-1))), 4) if _dg_dri1 is not None else ""
                    for _k, _v in (("dri_jerk", _jerk(_d_now, _dg_dri1, _dg_dri2)),
                                   ("swap_raw_jerk", _jerk(_dg_raw_now, _dg_raw1, _dg_raw2)),
                                   ("swap_out_jerk", _jerk(_o_now, _dg_out1, _dg_out2))):
                        if np.isfinite(_v):
                            _dg[_k] = round(float(_v), 4)
                    _dg["off_med"] = round(float(np.median(np.linalg.norm(
                        _o_now - _d_now, axis=-1))), 4)
                    _dg_dri2, _dg_dri1 = _dg_dri1, _d_now.copy()
                    _dg_raw2, _dg_raw1 = _dg_raw1, _dg_raw_now
                    _dg_out2, _dg_out1 = _dg_out1, _o_now.copy()
                except Exception as _e:
                    _dg["dri_jerk"] = "err"

            _sil_head_top = None
            _head_band = None

            # Driver mask warp: uniform scale around driver pelvis pixel by the
            # same factor force_height_scale, so mask matches the scaled skeleton.
            mask_xform = None
            if abs(s_eff - 1.0) > 1e-3 and mask_for_frame is not None:
                px = float(_anchor_px[0])
                py = float(_anchor_px[1])
                s_h = s_eff

                Hm, Wm = mask_for_frame.shape[:2]
                mask_u8 = (np.clip(mask_for_frame, 0.0, 1.0) * 255.0).astype(np.uint8)

                M = np.array([
                    [s_h, 0.0, px - s_h * px],
                    [0.0, s_h, py - s_h * py],
                ], dtype=np.float32)

                warped = cv2.warpAffine(
                    mask_u8, M, (Wm, Hm),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                mask_xform = warped.astype(np.float32) / 255.0
                _warp_frames += 1
                if _dg is not None:
                    _dg["warp"] = 1

            if mask_xform is None:
                if mask_for_frame is not None:
                    mask_xform = mask_for_frame.astype(np.float32)
                else:
                    Hf = int(driver_images.shape[1])
                    Wf = int(driver_images.shape[2])
                    mask_xform = np.ones((Hf, Wf), dtype=np.float32)

            # --- mask_from_swap: union the projected SWAP body silhouette into
            # the driver mask, so Wan gets room for a wider/bigger build than the
            # driver instead of being capped by the driver segmentation. Rows are
            # widened by the measured clothing ratios; the union is snapped to the
            # same 32px block grid as BlockifyMask upstream. At mask_from_swap=0
            # this whole block is a no-op (bit-exact old behavior).
            _sil_contours = None
            if (mask_from_swap > 0.0 and verts_swap is not None
                    and mask_for_frame is not None):
                try:
                    Hm2, Wm2 = mask_xform.shape[:2]
                    _v2d = _perspective_project(
                        verts_swap.astype(np.float64) + cam_t_swap[None, :], cam_int)
                    _sil = _splat_silhouette(_v2d, Hm2, Wm2, extra_dilate=2)
                    if _sil is not None:
                        if _cloth is not None and clothing_volume_strength > 0.0:
                            _yb = (
                                float((kp2d_swap[5, 1] + kp2d_swap[6, 1]) / 2.0),
                                float((kp2d_swap[_LHIP_IDX, 1] + kp2d_swap[_RHIP_IDX, 1]) / 2.0),
                                float((kp2d_swap[11, 1] + kp2d_swap[12, 1]) / 2.0),
                                float((kp2d_swap[13, 1] + kp2d_swap[14, 1]) / 2.0),
                            )
                            _rz = tuple(
                                1.0 + float(clothing_volume_strength) * (max(_cloth[_z], 1.0) - 1.0)
                                for _z in ("torso", "thigh", "shin"))
                            # Zone widening assumes shoulders above hips above
                            # knees. Sitting collapses hip and knee onto nearly
                            # the same row, the bands overlap and the widening
                            # smears across the chair. Verified: every sitting
                            # run had its hip zone rejected at 4.5-4.9 head-widths
                            # because the thighs were inside the hip band.
                            _hw_ref_px0 = float(np.linalg.norm(
                                np.asarray(kp2d_swap[3], dtype=np.float64)
                                - np.asarray(kp2d_swap[4], dtype=np.float64)))
                            if not np.isfinite(_hw_ref_px0) or _hw_ref_px0 < 4.0:
                                _hw_ref_px0 = 0.10 * float(_sil.shape[1])
                            _yb_ok = all(
                                (_yb[_i + 1] - _yb[_i]) > 0.45 * max(_hw_ref_px0, 1.0)
                                for _i in range(len(_yb) - 1))
                            if _yb_ok:
                                _sil = _widen_mask_rows(_sil, _yb, _rz)
                            else:
                                _sil = _widen_mask_rows(
                                    _sil, (_yb[0], _yb[-1]),
                                    (float(_cloth["torso"]),))
                                if i == 0:
                                    print("[BetaSwap] pose is not upright (height "
                                          "zones overlap) -> one uniform widening "
                                          "instead of per-zone, so the chair does "
                                          "not inherit leg slack")
                        # Row scaling moves an arm sideways; it never makes it
                        # thicker, which is why a bulkier reference so far only
                        # got its arms pushed apart. An isotropic dilation is the
                        # one operation that adds girth to a limb wherever it is,
                        # so the build ratio also drives a radius here. Erosion is
                        # allowed but kept small - the mask must not cut a body.
                        _hw_ref_px = float(np.linalg.norm(
                            np.asarray(kp2d_swap[3], dtype=np.float64)
                            - np.asarray(kp2d_swap[4], dtype=np.float64)))
                        if (np.isfinite(_hw_ref_px)
                                and abs(_vol_factor - 1.0) > 0.02
                                and _hw_ref_px > 4.0):
                            _rad = 0.35 * (_vol_factor - 1.0) * _hw_ref_px
                            _rad = float(min(max(_rad, -0.10 * _hw_ref_px),
                                             0.30 * _hw_ref_px))
                            _k = int(round(abs(_rad)))
                            if _k >= 1:
                                _ker = cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, (2 * _k + 1, 2 * _k + 1))
                                _sil = (cv2.dilate(_sil, _ker) if _rad > 0
                                        else cv2.erode(_sil, _ker))
                                if _dg is not None:
                                    _dg["limb_px"] = int(round(_rad))
                        _add = _sil.astype(np.float32) * float(mask_from_swap)
                        mask_xform = np.maximum(mask_xform, _add)
                        mask_xform = _block_snap_mask(mask_xform, block=32)
                        _sil_frames += 1
                        if _dg is not None:
                            _dg["sil"] = 1
                            _dg["sil_area"] = int((_sil > 0).sum())
                        try:
                            _es = float(np.linalg.norm(
                                np.asarray(kp2d_swap[3], dtype=np.float64)
                                - np.asarray(kp2d_swap[4], dtype=np.float64)))
                            _hw = max(_es * 1.2, 0.06 * Wm2)
                            _c0 = int(max(0, kp2d_swap[0][0] - _hw))
                            _c1 = int(min(Wm2, kp2d_swap[0][0] + _hw + 1))
                            _rows = np.any(_sil[:, _c0:_c1] > 0, axis=1)
                            _sil_head_top = (int(np.argmax(_rows))
                                             if _rows.any() else None)
                            _head_band = (_c0, _c1)
                        except Exception:
                            _sil_head_top, _head_band = None, None
                        _cn, _hier = cv2.findContours(
                            (_sil > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        _sil_contours = _cn
                        if _dg is not None:
                            try:
                                _sy, _sx = np.nonzero(_sil > 0)
                                if _sx.size:
                                    _dg["sil_x0"] = int(_sx.min())
                                    _dg["sil_x1"] = int(_sx.max())
                                    _dg["sil_y0"] = int(_sy.min())
                                    _dg["sil_y1"] = int(_sy.max())
                                _er = int(round(float(
                                    (kp2d_swap[3][1] + kp2d_swap[4][1]) / 2.0)))
                                if 0 <= _er < _sil.shape[0]:
                                    _cc = np.nonzero(_sil[_er] > 0)[0]
                                    if _cc.size:
                                        _dg["head_w_mesh"] = int(_cc.max() - _cc.min())
                            except Exception:
                                pass
                except Exception as _e:
                    print(f"[BetaSwap] mask_from_swap: frame {i} skipped "
                          f"({type(_e).__name__}: {_e})")

            # --- head clearance: with s_eff < 1 the swap head sits lower than
            # the driver head, exposing a band above it where stray driver hair
            # can live OUTSIDE the segmentation mask (observed: driver strands
            # popping over the generated head). Extend the mask upward in the
            # head columns by the drop distance so that band is regenerated.
            if s_eff < 0.999 and mask_for_frame is not None:
                try:
                    _nose = kp2d_dri[0]
                    _ear_half = float(np.linalg.norm(kp2d_dri[3] - kp2d_dri[4])) / 2.0
                    if not np.isfinite(_ear_half) or _ear_half < 4.0:
                        _ear_half = 0.075 * mask_xform.shape[1]
                    _c0 = int(max(0, _nose[0] - 2.0 * _ear_half))
                    _c1 = int(min(mask_xform.shape[1], _nose[0] + 2.0 * _ear_half))
                    _drop = (1.0 - s_eff) * max(float(_anchor_px[1] - _nose[1]), 0.0)
                    _up = int(min(np.ceil(_drop) + 12, 0.25 * mask_xform.shape[0]))
                    if _c1 > _c0 and _up > 0:
                        _ker = np.ones((_up, 1), dtype=np.uint8)
                        _sl = (mask_xform[:, _c0:_c1] > 0.5).astype(np.uint8)
                        _sl = cv2.dilate(_sl, _ker, anchor=(0, 0))
                        mask_xform[:, _c0:_c1] = np.maximum(
                            mask_xform[:, _c0:_c1], _sl.astype(np.float32))
                        mask_xform = _block_snap_mask(mask_xform, block=32)
                        _clear_frames += 1
                        if _dg is not None:
                            _dg["clear"] = 1
                            _dg["clear_up_px"] = int(_up)
                except Exception as _e:
                    print(f"[BetaSwap] head clearance: frame {i} skipped "
                          f"({type(_e).__name__}: {_e})")

            # --- headwear reserve: the mask is the only thing that decides whether
            # a hat / headband / ears can exist at all, and nothing else in this
            # node knows the character wears any. Open the head columns upward by
            # the amount measured on the reference, every frame, unconditionally.
            if _hw_rise > 0.02 and mask_xform is not None:
                try:
                    _es_o = float(np.linalg.norm(
                        np.asarray(kp2d_swap[3], dtype=np.float64)
                        - np.asarray(kp2d_swap[4], dtype=np.float64)))
                    if not np.isfinite(_es_o) or _es_o < 4.0:
                        _es_o = 0.15 * mask_xform.shape[1]
                    _up2 = int(min(np.ceil(_hw_rise * _es_o) + 8,
                                   0.45 * mask_xform.shape[0]))
                    _hb = max(_hw_half * _es_o, 0.9 * _es_o)
                    _nx = float(kp2d_swap[0][0])
                    _d0 = int(max(0, _nx - _hb))
                    _d1 = int(min(mask_xform.shape[1], _nx + _hb + 1))
                    if _up2 > 0 and _d1 > _d0:
                        # Dilating the band upward lifted every foreground pixel
                        # in it, shoulders included, so the reserve came out as a
                        # wide slab. Draw the room the accessory actually needs:
                        # an ellipse seated on the crown, which tapers instead of
                        # squaring off the sky above the character.
                        _ht_r = _dg.get("head_top") if _dg is not None else None
                        if _ht_r is None:
                            _ht_r = float(np.min(np.nonzero(
                                np.any(mask_xform[:, _d0:_d1] > 0.5, axis=1))[0]))
                        _cy = int(round(float(_ht_r)))
                        _add = np.zeros_like(mask_xform, dtype=np.uint8)
                        cv2.ellipse(_add, (int(round(_nx)), _cy),
                                    (int(max(_hb, 4)), int(max(_up2, 2))),
                                    0, 180, 360, 1, -1)
                        _add[_cy + 1:, :] = 0
                        mask_xform = np.maximum(mask_xform,
                                                _add.astype(np.float32))
                        mask_xform = _block_snap_mask(mask_xform, block=32)
                        _hw_frames += 1
                        if _dg is not None:
                            _dg["hw_up_px"] = int(_up2)
                            _dg["hw_half_px"] = int(_hb)
                except Exception as _e:
                    print(f"[BetaSwap] headwear reserve: frame {i} skipped "
                          f"({type(_e).__name__}: {_e})")

            # --- mask budget. Every complaint in this batch - a redrawn chair,
            # the reference backdrop appearing behind the character, white
            # invented between the legs - lands on one number: how much bigger
            # the conditioning mask is than the body it is supposed to cover.
            # Measured across 45 runs, mask/silhouette sat at 1.39-1.45 while
            # nothing was invented, and every run the user flagged is above 1.9
            # (2.65, 2.41, 2.39, 2.43). Room the character does not fill is room
            # the model fills with something else, so cap it and pay for the
            # overrun out of the discretionary parts by eroding back.
            if mask_xform is not None and _sil is not None:
                try:
                    _sa = float((_sil > 0).sum())
                    _ma = float((mask_xform > 0.5).sum())
                    if _sa > 100.0 and _ma > _MASK_BUDGET * _sa:
                        _need = _ma - _MASK_BUDGET * _sa
                        _per = max(float(np.sqrt(_sa)), 1.0)
                        _kk = int(min(max(round(_need / (4.0 * _per)), 1), 24))
                        _ker = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (2 * _kk + 1, 2 * _kk + 1))
                        _tr = cv2.erode((mask_xform > 0.5).astype(np.uint8), _ker)
                        # never erode below the body itself
                        _tr = np.maximum(_tr, (_sil > 0).astype(np.uint8))
                        mask_xform = _block_snap_mask(
                            _tr.astype(np.float32), block=32)
                        _trim_frames += 1
                        if _dg is not None:
                            _dg["trim_px"] = int(_kk)
                    if _dg is not None:
                        _dg["mask_over_sil"] = round(
                            float((mask_xform > 0.5).sum()) / max(_sa, 1.0), 4)
                except Exception as _e:
                    print(f"[BetaSwap] mask budget: frame {i} skipped ({_e})")

            # --- temporal_smooth: decaying union of the mask over frames.
            # The union is snapped to a 32px grid, so a border block that sits
            # near the threshold switches fully on/off between frames - Wan sees
            # the conditioning region blink, which reads as micro-stutter. A
            # block that was on keeps a decaying residual instead of dropping
            # instantly; below 0.5 it is released. Widen-only, so this can never
            # shrink the region below what this frame asked for.
            _dg_mask_pre = (np.asarray(mask_xform, dtype=np.float32).copy()
                            if _dg is not None else None)
            if _ts > 0.0:
                _cur = np.asarray(mask_xform, dtype=np.float32)
                # Everything feeding this mask is binary (upstream BlockifyMask,
                # our own 32px snap), and everything downstream treats it as a
                # region, not as an alpha. An earlier version let held blocks
                # sit at 0.6/0.36 while they decayed - a partial-strength
                # conditioning region, i.e. a washed-out patch. The hold now
                # only decides WHETHER a block stays on; the value written is
                # the same 1.0 as any other on-block. If the incoming mask is
                # genuinely soft (someone fed a feathered matte), the hold is
                # skipped entirely rather than binarising their data.
                _midfrac = float(np.mean((_cur > 0.02) & (_cur < 0.98)))
                if _midfrac <= 0.01:
                    if _hold is None or _hold.shape != _cur.shape:
                        _hold = _cur.copy()
                    else:
                        _hold = np.maximum(_cur, _hold * _ts)
                    mask_xform = np.maximum(
                        _cur, (_hold >= 0.5).astype(np.float32))
                else:
                    _hold = None
                    mask_xform = _cur

            if _dg is not None:
                try:
                    _dg["mask_area"] = int((mask_xform > 0.5).sum())
                    _mm = np.asarray(mask_xform, dtype=np.float32)
                    _dg["mask_frac"] = round(float(np.mean(
                        (_mm > 0.02) & (_mm < 0.98))), 6)
                    if _head_band is not None:
                        _c0, _c1 = _head_band
                        _r = np.any(_mm[:, _c0:_c1] > 0.5, axis=1)
                        if _r.any():
                            _mtop = int(np.argmax(_r))
                            _dg["mask_top"] = _mtop
                            _htop = _sil_head_top
                            if _htop is None:
                                _em = (np.asarray(kp2d_swap[1], dtype=np.float64)
                                       + np.asarray(kp2d_swap[2], dtype=np.float64)) / 2.0
                                _htop = int(_em[1] - 1.6 * abs(
                                    float(kp2d_swap[0][1]) - float(_em[1])))
                            _dg["head_top"] = int(_htop)
                            _dg["headroom_px"] = int(_htop - _mtop)
                            _dg["headroom_blk"] = round((_htop - _mtop) / 32.0, 2)
                    _bfp = _block_flips(_dg_mask_pre, _dg_maskpre_prev)
                    _bfq = _block_flips(mask_xform, _dg_mask_prev)
                    if _bfp >= 0:
                        _dg["blocks_flip_pre"] = _bfp
                    if _bfq >= 0:
                        _dg["blocks_flip_post"] = _bfq
                    _dg_maskpre_prev = _dg_mask_pre
                    _dg_mask_prev = np.asarray(mask_xform, dtype=np.float32).copy()
                except Exception:
                    pass

            transformed_masks_out.append(mask_xform)


            H_src, W_src = dri_bgr.shape[:2]
            if _dg is not None:
                try:
                    _ks = np.asarray(kp2d_swap, dtype=np.float64)
                    _dg["ear_draw"] = round(float(np.linalg.norm(
                        _ks[3] - _ks[4])), 3)
                    _dg["sh_draw"] = round(float(np.linalg.norm(
                        _ks[5] - _ks[6])), 3)
                    _dg["hip_draw"] = round(float(np.linalg.norm(
                        _ks[_LHIP_IDX] - _ks[_RHIP_IDX])), 3)
                    # The bare-mesh crown scaled by the same factor the head
                    # keypoints were scaled by: an estimate of where the crown of
                    # the head Wan will actually render ends up.
                    # The head is no longer rescaled, so the drawn crown IS the
                    # mesh crown. What matters now is whether the mask holds the
                    # accessory: needed vs available room above that crown.
                    _ht0 = _dg.get("head_top")
                    _mt0 = _dg.get("mask_top")
                    if _ht0 is not None:
                        _dg["crown_est"] = int(round(float(_ht0)))
                        if _mt0 is not None:
                            _dg["headroom_pose_px"] = int(round(
                                float(_ht0) - float(_mt0)))
                    if _hw_rise > 0.02:
                        _need = _hw_rise * float(np.linalg.norm(_ks[3] - _ks[4]))
                        _dg["hw_need_px"] = int(round(_need))
                        if _ht0 is not None and _mt0 is not None:
                            _dg["hw_deficit_px"] = int(round(
                                _need - (float(_ht0) - float(_mt0))))
                except Exception:
                    pass

            meta_src = _build_aapose_meta(kp2d_swap, H_src, W_src)

            # Driver hands postfix substitution (always on)
            lhand_xy, lhand_conf, rhand_xy, rhand_conf = \
                _extract_driver_hands_from_pose_data(
                    driver_pose_data, i, H_src, W_src)
            # Attach driver hands to the SWAPPED wrists: keep driver finger
            # articulation, but translate the whole hand by
            # (swap_wrist_px - driver_hand_wrist_px). Hand point 0 is the wrist
            # (MHR 62 = LWrist, MHR 41 = RWrist). Identity proportions -> delta~0,
            # so behavior is unchanged for shape_strength=0.
            if lhand_xy is not None and lhand_xy.shape[0] == 21:
                _hs_l = _forearm_ratio(kp2d_swap, kp2d_dri, 7, 62)
                if _dg is not None:
                    _dg["hand_l"] = round(float(_hs_l), 4)
                lhand_xy = kp2d_swap[62].astype(np.float32) + (
                    (lhand_xy - lhand_xy[0]) * _hs_l
                )
            if rhand_xy is not None and rhand_xy.shape[0] == 21:
                _hs_r = _forearm_ratio(kp2d_swap, kp2d_dri, 8, 41)
                if _dg is not None:
                    _dg["hand_r"] = round(float(_hs_r), 4)
                rhand_xy = kp2d_swap[41].astype(np.float32) + (
                    (rhand_xy - rhand_xy[0]) * _hs_r
                )
            if lhand_xy is not None and lhand_xy.shape[0] == 21:
                meta_src.kps_lhand = lhand_xy.astype(np.float32)
                meta_src.kps_lhand_p = lhand_conf.astype(np.float32) \
                    if lhand_conf is not None else np.full(21, 1.0, dtype=np.float32)
                hands_substituted += 1
            if rhand_xy is not None and rhand_xy.shape[0] == 21:
                meta_src.kps_rhand = rhand_xy.astype(np.float32)
                meta_src.kps_rhand_p = rhand_conf.astype(np.float32) \
                    if rhand_conf is not None else np.full(21, 1.0, dtype=np.float32)

            face_xy = None
            face_p = None
            if draw_face:
                face_xy, face_p = _extract_driver_face_from_pose_data(
                    driver_pose_data, i, H_src, W_src,
                )
                if face_xy is not None and face_shape_strength > 0.0:
                    face_xy = _scale_face_to_swap(
                        face_xy, _head_measured, kp2d_dri,
                        float(face_shape_strength), _face_ema,
                        ema_alpha=max(0.5, _ts))
                if face_xy is not None:
                    meta_src.kps_face = face_xy
                    meta_src.kps_face_p = face_p
                    face_rendered_count += 1

            canvas_src = np.zeros((H_src, W_src, 3), dtype=np.uint8)
            pose_src = draw_aapose_by_meta_new(
                canvas_src, meta_src,
                draw_hand=draw_hand, draw_head=True,
                body_stick_width=eff_body_stick_width,
                hand_stick_width=hand_stick_width,
                threshold=0.3,
            )

            if draw_face and face_xy is not None:
                face_thickness = max(2, int(round(max(H_src, W_src) / 360)))
                pose_src = _draw_face_with_conf_gate(
                    pose_src, face_xy, face_p,
                    thickness=face_thickness, threshold=FACE_CONF_GATE,
                )

            pose_out = padding_resize(pose_src, target_height, target_width)
            if _dg is not None:
                try:
                    _pk = np.asarray(pose_out)
                    _nzm = _pk.any(axis=2) if _pk.ndim == 3 else (_pk > 0)
                    _pys, _pxs = np.nonzero(_nzm)
                    if _pxs.size:
                        _dg["pose_x0"] = int(_pxs.min())
                        _dg["pose_x1"] = int(_pxs.max())
                        _dg["pose_y0"] = int(_pys.min())
                        _dg["pose_y1"] = int(_pys.max())
                        _dg["pose_w"] = int(_pxs.max() - _pxs.min())
                        _dg["pose_h"] = int(_pys.max() - _pys.min())
                    _dg["pose_ink"] = int(_nzm.sum())
                except Exception:
                    pass
            pose_images_out.append(pose_out)


            dri_rgb = cv2.cvtColor(dri_bgr, cv2.COLOR_BGR2RGB)
            overlay = draw_aapose_by_meta_new(
                dri_rgb.copy(), meta_src,
                draw_hand=draw_hand, draw_head=True,
                body_stick_width=eff_body_stick_width,
                hand_stick_width=hand_stick_width,
                threshold=0.3,
            )
            if draw_face and face_xy is not None:
                face_thickness = max(2, int(round(max(H_src, W_src) / 360)))
                overlay = _draw_face_with_conf_gate(
                    overlay, face_xy, face_p,
                    thickness=face_thickness, threshold=FACE_CONF_GATE,
                )
            if _sil_contours is not None:
                try:
                    cv2.drawContours(overlay, _sil_contours, -1, (255, 220, 0), 1)
                except Exception:
                    pass
            debug_overlay_out.append(overlay)

            if _dg is not None:
                if _face_ema.get("sx") is not None:
                    _dg["face_sx"] = round(float(_face_ema["sx"]), 4)
                    _dg["face_sy"] = round(float(_face_ema["sy"]), 4)
                _diag_rows.append(_dg)

            if pbar is not None:
                pbar.update(1)

        pose_np = np.stack(pose_images_out, axis=0).astype(np.float32) / 255.0
        pose_tensor = torch.from_numpy(pose_np)

        shapes = {o.shape for o in debug_overlay_out}
        if len(shapes) == 1:
            dbg_np = np.stack(debug_overlay_out, axis=0).astype(np.float32) / 255.0
        else:
            ref_shape = debug_overlay_out[0].shape[:2]
            dbg_np = np.stack(
                [cv2.resize(o, (ref_shape[1], ref_shape[0])) for o in debug_overlay_out],
                axis=0,
            ).astype(np.float32) / 255.0
        dbg_tensor = torch.from_numpy(dbg_np)

        mshapes = {m.shape for m in transformed_masks_out}
        if len(mshapes) == 1:
            mask_np = np.stack(transformed_masks_out, axis=0).astype(np.float32)
        else:
            ref_mshape = transformed_masks_out[0].shape
            mask_np = np.stack(
                [cv2.resize(m, (ref_mshape[1], ref_mshape[0]),
                            interpolation=cv2.INTER_NEAREST)
                 for m in transformed_masks_out],
                axis=0,
            ).astype(np.float32)
        mask_np = np.clip(mask_np, 0.0, 1.0)
        mask_tensor = torch.from_numpy(mask_np)


        print(f"[BetaSwap] Driver hands: {hands_substituted} hands replaced over {N} frames")
        if draw_face:
            print(f"[BetaSwap] Face rendering: {face_rendered_count}/{N} frames")
        if use_keypoint_prompts:
            avg = (prompts_used_total / max(1, prompts_frames_with))
            print(f"[BetaSwap] Keypoint prompts: {prompts_frames_with}/{N} frames refined, "
                  f"avg {avg:.1f} prompts/frame")
        if _warp_frames > 0:
            print(f"[BetaSwap] Driver mask warped on {_warp_frames}/{N} frames "
                  f"(last s_eff={_s_last:.3f})")
        if auto_height_from_reference > 0.0 and _ah_ratio is not None:
            print(f"[BetaSwap] auto_height: swap/driver skeletal ratio (EMA) = "
                  f"{_ah_ratio:.3f} -> last s_eff={_s_last:.3f} "
                  f"(chain={_chain_state}, manual trim "
                  f"force_height_scale={force_height_scale:.2f})")
        if _trim_frames > 0:
            print(f"[BetaSwap] mask budget: trimmed back on {_trim_frames}/{N} "
                  f"frames to stay under {_MASK_BUDGET:.2f}x the body silhouette")
        if _hw_frames > 0:
            print(f"[BetaSwap] headwear: mask opened above the head on "
                  f"{_hw_frames}/{N} frames ({_hw_rise:.2f} ear-spans up, "
                  f"{_hw_half:.2f} sideways, measured on the reference)")
        if _clear_frames > 0:
            print(f"[BetaSwap] head clearance: mask extended upward over the head "
                  f"on {_clear_frames}/{N} frames (s_eff<1 drop compensation)")
        if _anchor_face_frames > 0:
            print(f"[BetaSwap] anchor: FACE (nose) on {_anchor_face_frames}/{N} "
                  f"frames (driver pelvis off-frame), pelvis on the rest")
        if _sil_frames > 0:
            print(f"[BetaSwap] mask_from_swap: swap-body silhouette unioned into mask "
                  f"on {_sil_frames}/{N} frames (32px block-snapped)")
        if _smooth_frames > 0:
            print(f"[BetaSwap] temporal_smooth={_ts:.2f}: offset-field EMA on "
                  f"{_smooth_frames}/{N} frames (driver motion passes 1:1), "
                  f"intrinsics EMA + decaying mask union active")
        if draw_face and face_shape_strength > 0.0 and _face_ema.get("sx") is not None:
            print(f"[BetaSwap] face_shape: driver dlib68 scaled to swap head "
                  f"(last sx={_face_ema['sx']:.3f}, sy={_face_ema['sy']:.3f})")
        if diagnostics and _diag_rows:
            _diag_write_report(_diag_rows, None, log=_dlog)
            _dlog.table(_diag_rows)

        print(f"[BetaSwap] Done: {N} frames -> pose_images {tuple(pose_tensor.shape)}, "
              f"debug_overlay {tuple(dbg_tensor.shape)}, "
              f"transformed_driver_mask {tuple(mask_tensor.shape)}")

        return (pose_tensor, dbg_tensor, mask_tensor)

def _draw_face_with_conf_gate(img, kps_face, kps_face_p, thickness=2, threshold=0.5):
    img = img.copy() if not img.flags.writeable else img
    for key, item in FACE_CUSTOM_STYLE.items():
        idxs = item["indexs"]
        confs = np.array([kps_face_p[i] for i in idxs])
        if (confs > threshold).sum() < max(2, len(idxs) // 2):
            continue
        pts = np.array([kps_face[i] for i in idxs], dtype=np.int32)
        connect = item.get("connect", True)
        color = item["color"]
        close = item.get("close", False)
        if connect:
            cv2.polylines(img, [pts], close, color, thickness=thickness, lineType=cv2.LINE_AA)
        else:
            r = max(1, thickness * 2)
            for p in pts:
                cv2.circle(img, tuple(int(v) for v in p), r, color=color, thickness=-1)
    return img

NODE_CLASS_MAPPINGS = {
    "SAM3DBodyBetaSwapPoseRender": SAM3DBodyBetaSwapPoseRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3DBodyBetaSwapPoseRender": "SAM 3D Body: beta-Swap Pose Render (Wan Animate)",
}