#!/usr/bin/env python3
"""
encode.py — 多尺度自适应VQ图像编码与解码
=========================================

自适应编码策略（五遍编码）：
1. 先用 8×8 码本编码整图
2. PSNR < 阈值的块 → 拆为 4 个 4×4 块重新编码
3. 仍不满足阈值的 4×4 块 → 尝试 2×4/4×2 方向分割，选最优方向
4. 仍不满足阈值 → 拆为 4 个 2×2 块编码
5. 仍不满足阈值的 2×2 块 → 尝试 1×2/2×1 方向分割

使用方式：
    python encode.py img_1.png
    python encode.py --psnr 35 img_1.png
    python encode.py --psnr 30 --all
    python encode.py --decode result.npz --ref img_1.png
"""

import os, sys, time, math, glob, argparse, re
import numpy as np
import cv2
from PIL import Image, ImageFilter


WORK_DIR = os.path.dirname(os.path.abspath(__file__))


# ==================== 码本加载 ====================

def load_codebook(npz_path):
    data = np.load(npz_path)
    cb = data['codebook'].astype(np.float32)
    K = int(data['K'])
    if 'block_h' in data:
        bh = int(data['block_h'])
        bw = int(data['block_w'])
    else:
        bs = int(data['block_size']) if 'block_size' in data else 8
        bh, bw = bs, bs
    return cb, K, bh, bw


def find_codebooks(cb_dir=None):
    d = cb_dir or WORK_DIR
    paths = {}
    for fn in os.listdir(d):
        if not fn.startswith('codebook_S') or not fn.endswith('.npz'):
            continue
        m = re.match(r'codebook_S(\d+(?:x\d+)?)_K(\d+)\.npz', fn)
        if m:
            scale_str = m.group(1)
            if 'x' in scale_str:
                h, w = scale_str.split('x')
                key = (int(h), int(w))
            else:
                s = int(scale_str)
                key = (s, s)
            paths[key] = os.path.join(d, fn)
    return paths


def load_all_codebooks(cb_paths):
    codebooks = {}
    for key in cb_paths:
        path = cb_paths[key]
        cb, K, bh, bw = load_codebook(path)
        assert (bh, bw) == key, f"{path} 中 block=({bh},{bw}), 期望 {key}"
        codebooks[key] = (cb, K)
        display = f'{bh}×{bw}' if bh != bw else f'{bh}×{bw}'
        print(f"[码本] {display}: {os.path.basename(path)}, K={K}, shape={cb.shape}")
    return codebooks


# ==================== 工具函数 ====================

def batch_nearest(blocks_flat, cb):
    CB_sq = np.sum(cb ** 2, axis=1)
    indices = []
    batch_sz = 2000
    for i in range(0, len(blocks_flat), batch_sz):
        batch = blocks_flat[i:i + batch_sz]
        X_sq = np.sum(batch ** 2, axis=1)
        cross = batch @ cb.T
        dists = X_sq[:, None] + CB_sq[None, :] - 2 * cross
        indices.extend(np.argmin(dists, axis=1).tolist())
    return np.array(indices, dtype=np.int32)


def vec_block_psnr(orig_blocks, recon_blocks):
    diff = orig_blocks.astype(np.float32) - recon_blocks.astype(np.float32)
    mse = np.mean(diff ** 2, axis=tuple(range(1, diff.ndim)))
    psnr = np.where(mse < 1e-8, 100.0,
                    20.0 * np.log10(255.0 / np.sqrt(np.maximum(mse, 1e-8))))
    return psnr


def compute_psnr(orig, recon):
    mse = np.mean((orig.astype(np.float32) - recon.astype(np.float32)) ** 2)
    psnr = 20 * math.log10(255.0 / math.sqrt(max(mse, 1e-8)))
    return psnr, mse


# ==================== 自适应编码 ====================

def encode_image_adaptive(pixels, codebooks, psnr_thr):
    """
    五遍自适应编码：
      Pass 1 — 全图 8×8
      Pass 2 — 不满足阈值的 8×8 块 → 4×4
      Pass 3 — 不满足阈值的 4×4 块 → 2×4/4×2 (选最优方向)
      Pass 4 — 不满足阈值的 → 2×2
      Pass 5 — 不满足阈值的 2×2 块 → 1×2/2×1 (选最优方向)
    """
    H, W = pixels.shape[:2]
    ph = (8 - H % 8) % 8
    pw = (8 - W % 8) % 8
    if ph or pw:
        pixels = np.pad(pixels, ((0, ph), (0, pw), (0, 0)), mode='edge')
    HP, WP = pixels.shape[:2]
    H8, W8 = HP // 8, WP // 8

    has_rect4 = (2, 4) in codebooks and (4, 2) in codebooks
    has_rect2 = (1, 2) in codebooks and (2, 1) in codebooks

    cb8, K8 = codebooks[(8, 8)]
    cb4, K4 = codebooks[(4, 4)]
    cb2, K2 = codebooks[(2, 2)]
    cb_2x4 = cb_4x2 = cb_1x2 = cb_2x1 = None
    K_2x4 = K_4x2 = K_1x2 = K_2x1 = 0
    if has_rect4:
        cb_2x4, K_2x4 = codebooks[(2, 4)]
        cb_4x2, K_4x2 = codebooks[(4, 2)]
    if has_rect2:
        cb_1x2, K_1x2 = codebooks[(1, 2)]
        cb_2x1, K_2x1 = codebooks[(2, 1)]

    # ---------- Pass 1 : 8×8 ----------
    blk8 = []
    for r in range(0, HP, 8):
        for c in range(0, WP, 8):
            blk8.append(pixels[r:r + 8, c:c + 8].astype(np.float32).flatten())
    blk8 = np.array(blk8)

    idx8_all = batch_nearest(blk8, cb8)
    rec8_all = cb8[idx8_all].reshape(-1, 8, 8, 3)
    psnr8 = vec_block_psnr(blk8.reshape(-1, 8, 8, 3), rec8_all)

    need_finer_8 = psnr8 < psnr_thr
    sub8to4 = need_finer_8.reshape(H8, W8)
    idx_s8 = idx8_all[~need_finer_8]

    n8_pass = int(np.sum(~need_finer_8))
    n8_fail = int(np.sum(need_finer_8))
    print(f"    Pass 1 (8×8): {len(blk8)} 块 | "
          f"{n8_pass} 通过 | {n8_fail} 需细分 → 4×4")

    # ---------- Pass 2 : 4×4 ----------
    sub4_type = np.zeros((H8, W8, 2, 2), dtype=np.int8)
    idx_s4_list = []
    fail4_info = []

    fail8 = np.where(need_finer_8)[0]
    if len(fail8) > 0:
        blk4_list = []
        blk4_map = []
        for i8 in fail8:
            r8 = (i8 // W8) * 8
            c8 = (i8 % W8) * 8
            for sr in range(2):
                for sc in range(2):
                    r4 = r8 + sr * 4
                    c4 = c8 + sc * 4
                    blk4_list.append(
                        pixels[r4:r4 + 4, c4:c4 + 4].astype(np.float32).flatten())
                    blk4_map.append((i8, sr, sc))

        blk4 = np.array(blk4_list)
        idx4_all = batch_nearest(blk4, cb4)
        rec4_all = cb4[idx4_all].reshape(-1, 4, 4, 3)
        psnr4 = vec_block_psnr(blk4.reshape(-1, 4, 4, 3), rec4_all)

        need_finer_4 = psnr4 < psnr_thr

        for k, (i8, sr, sc) in enumerate(blk4_map):
            r8g = i8 // W8
            c8g = i8 % W8
            if not need_finer_4[k]:
                idx_s4_list.append(idx4_all[k])
                sub4_type[r8g, c8g, sr, sc] = 0
            else:
                fail4_info.append((i8, sr, sc))

    idx_s4 = np.array(idx_s4_list, dtype=np.int32)
    n4_pass = len(idx_s4_list)
    n4_fail = len(fail4_info)
    print(f"    Pass 2 (4×4): {n4_pass + n4_fail} 块 | "
          f"{n4_pass} 通过 | {n4_fail} 需细分")

    # ---------- Pass 3 : 2×4 / 4×2 ----------
    idx_rect4_list = []
    fail4to2_info = []

    if has_rect4 and len(fail4_info) > 0:
        n_fail4 = len(fail4_info)
        all_24_top, all_24_bot = [], []
        all_42_left, all_42_right = [], []
        orig_4x4_list = []

        for i8, sr, sc in fail4_info:
            r8 = (i8 // W8) * 8
            c8 = (i8 % W8) * 8
            r4 = r8 + sr * 4
            c4 = c8 + sc * 4
            orig = pixels[r4:r4 + 4, c4:c4 + 4].astype(np.float32)
            orig_4x4_list.append(orig)
            all_24_top.append(orig[0:2, 0:4].flatten())
            all_24_bot.append(orig[2:4, 0:4].flatten())
            all_42_left.append(orig[0:4, 0:2].flatten())
            all_42_right.append(orig[0:4, 2:4].flatten())

        orig_4x4_arr = np.array(orig_4x4_list)

        blk_24 = np.array(all_24_top + all_24_bot)
        idx_24 = batch_nearest(blk_24, cb_2x4)
        idx_24_top = idx_24[:n_fail4]
        idx_24_bot = idx_24[n_fail4:]
        rec_24_top = cb_2x4[idx_24_top].reshape(-1, 2, 4, 3)
        rec_24_bot = cb_2x4[idx_24_bot].reshape(-1, 2, 4, 3)
        rec_24_full = np.concatenate([rec_24_top, rec_24_bot], axis=1)
        psnr_24 = vec_block_psnr(orig_4x4_arr, rec_24_full)

        blk_42 = np.array(all_42_left + all_42_right)
        idx_42 = batch_nearest(blk_42, cb_4x2)
        idx_42_left = idx_42[:n_fail4]
        idx_42_right = idx_42[n_fail4:]
        rec_42_left = cb_4x2[idx_42_left].reshape(-1, 4, 2, 3)
        rec_42_right = cb_4x2[idx_42_right].reshape(-1, 4, 2, 3)
        rec_42_full = np.concatenate([rec_42_left, rec_42_right], axis=2)
        psnr_42 = vec_block_psnr(orig_4x4_arr, rec_42_full)

        for k, (i8, sr, sc) in enumerate(fail4_info):
            r8g = i8 // W8
            c8g = i8 % W8
            if psnr_24[k] >= psnr_42[k]:
                best_psnr, best_dir = psnr_24[k], 1
                best_idx = [idx_24_top[k], idx_24_bot[k]]
            else:
                best_psnr, best_dir = psnr_42[k], 2
                best_idx = [idx_42_left[k], idx_42_right[k]]

            if best_psnr >= psnr_thr:
                sub4_type[r8g, c8g, sr, sc] = best_dir
                idx_rect4_list.extend(best_idx)
            else:
                sub4_type[r8g, c8g, sr, sc] = 3
                fail4to2_info.append((i8, sr, sc))

        n_rect4 = len(idx_rect4_list) // 2
        print(f"    Pass 3 (2×4/4×2): {n_fail4} 块 | "
              f"{n_rect4} rect通过 | {len(fail4to2_info)} → 2×2")
    else:
        for i8, sr, sc in fail4_info:
            r8g = i8 // W8
            c8g = i8 % W8
            sub4_type[r8g, c8g, sr, sc] = 3
            fail4to2_info.append((i8, sr, sc))
        print(f"    Pass 3: 跳过(无rect码本) | "
              f"{len(fail4to2_info)} 块直入2×2")

    idx_rect4 = np.array(idx_rect4_list, dtype=np.int32)

    # ---------- Pass 4 : 2×2 ----------
    sub2_type = np.zeros((H8, W8, 2, 2, 2, 2), dtype=np.int8)
    idx_s2_list = []
    fail2_info = []
    idx2_all_ref = {}

    if len(fail4to2_info) > 0:
        blk2_list = []
        blk2_map = []
        for i8, sr, sc in fail4to2_info:
            r8 = (i8 // W8) * 8
            c8 = (i8 % W8) * 8
            r4 = r8 + sr * 4
            c4 = c8 + sc * 4
            for ssr in range(2):
                for ssc in range(2):
                    r2 = r4 + ssr * 2
                    c2 = c4 + ssc * 2
                    blk2_list.append(
                        pixels[r2:r2 + 2, c2:c2 + 2].astype(np.float32).flatten())
                    blk2_map.append((i8, sr, sc, ssr, ssc))

        blk2 = np.array(blk2_list)
        idx2_all = batch_nearest(blk2, cb2)
        rec2_all = cb2[idx2_all].reshape(-1, 2, 2, 3)
        psnr2 = vec_block_psnr(blk2.reshape(-1, 2, 2, 3), rec2_all)

        need_finer_2 = psnr2 < psnr_thr

        for k, (i8, sr, sc, ssr, ssc) in enumerate(blk2_map):
            r8g = i8 // W8
            c8g = i8 % W8
            if not need_finer_2[k]:
                idx_s2_list.append(idx2_all[k])
                sub2_type[r8g, c8g, sr, sc, ssr, ssc] = 0
            else:
                if has_rect2:
                    fail2_info.append((i8, sr, sc, ssr, ssc, k))
                    idx2_all_ref[k] = idx2_all[k]
                else:
                    idx_s2_list.append(idx2_all[k])
                    sub2_type[r8g, c8g, sr, sc, ssr, ssc] = 0

    idx_s2 = np.array(idx_s2_list, dtype=np.int32)
    n2_pass = len(idx_s2_list)
    n2_fail = len(fail2_info)
    print(f"    Pass 4 (2×2): {n2_pass + n2_fail} 块 | "
          f"{n2_pass} 通过 | {n2_fail} 需细分")

    # ---------- Pass 5 : 1×2 / 2×1 ----------
    idx_rect2_list = []

    if has_rect2 and len(fail2_info) > 0:
        n_fail2 = len(fail2_info)
        all_12_top, all_12_bot = [], []
        all_21_left, all_21_right = [], []
        orig_2x2_list = []

        for i8, sr, sc, ssr, ssc, k2 in fail2_info:
            r8 = (i8 // W8) * 8
            c8 = (i8 % W8) * 8
            r4 = r8 + sr * 4
            c4 = c8 + sc * 4
            r2 = r4 + ssr * 2
            c2 = c4 + ssc * 2
            orig = pixels[r2:r2 + 2, c2:c2 + 2].astype(np.float32)
            orig_2x2_list.append(orig)
            all_12_top.append(orig[0:1, 0:2].flatten())
            all_12_bot.append(orig[1:2, 0:2].flatten())
            all_21_left.append(orig[0:2, 0:1].flatten())
            all_21_right.append(orig[0:2, 1:2].flatten())

        orig_2x2_arr = np.array(orig_2x2_list)

        blk_12 = np.array(all_12_top + all_12_bot)
        idx_12 = batch_nearest(blk_12, cb_1x2)
        idx_12_top = idx_12[:n_fail2]
        idx_12_bot = idx_12[n_fail2:]
        rec_12_top = cb_1x2[idx_12_top].reshape(-1, 1, 2, 3)
        rec_12_bot = cb_1x2[idx_12_bot].reshape(-1, 1, 2, 3)
        rec_12_full = np.concatenate([rec_12_top, rec_12_bot], axis=1)
        psnr_12 = vec_block_psnr(orig_2x2_arr, rec_12_full)

        blk_21 = np.array(all_21_left + all_21_right)
        idx_21 = batch_nearest(blk_21, cb_2x1)
        idx_21_left = idx_21[:n_fail2]
        idx_21_right = idx_21[n_fail2:]
        rec_21_left = cb_2x1[idx_21_left].reshape(-1, 2, 1, 3)
        rec_21_right = cb_2x1[idx_21_right].reshape(-1, 2, 1, 3)
        rec_21_full = np.concatenate([rec_21_left, rec_21_right], axis=2)
        psnr_21 = vec_block_psnr(orig_2x2_arr, rec_21_full)

        for k, (i8, sr, sc, ssr, ssc, k2) in enumerate(fail2_info):
            r8g = i8 // W8
            c8g = i8 % W8
            if psnr_12[k] >= psnr_21[k]:
                sub2_type[r8g, c8g, sr, sc, ssr, ssc] = 1
                idx_rect2_list.extend([idx_12_top[k], idx_12_bot[k]])
            else:
                sub2_type[r8g, c8g, sr, sc, ssr, ssc] = 2
                idx_rect2_list.extend([idx_21_left[k], idx_21_right[k]])

        n_rect2 = len(idx_rect2_list) // 2
        print(f"    Pass 5 (1×2/2×1): {n2_fail} 块 → {n_rect2} rect编码")
    else:
        print(f"    Pass 5: 跳过(无rect2码本)")

    idx_rect2 = np.array(idx_rect2_list, dtype=np.int32)

    return {
        'version': 2,
        'H': H, 'W': W, 'HP': HP, 'WP': WP,
        'H8': H8, 'W8': W8,
        'K8': K8, 'K4': K4, 'K2': K2,
        'K_2x4': K_2x4, 'K_4x2': K_4x2,
        'K_1x2': K_1x2, 'K_2x1': K_2x1,
        'has_rect4': has_rect4, 'has_rect2': has_rect2,
        'sub8to4': sub8to4,
        'sub4_type': sub4_type,
        'sub2_type': sub2_type,
        'idx_s8': idx_s8,
        'idx_s4': idx_s4,
        'idx_rect4': idx_rect4,
        'idx_s2': idx_s2,
        'idx_rect2': idx_rect2,
        'n_s8': len(idx_s8),
        'n_s4': n4_pass,
        'n_rect4': len(idx_rect4_list) // 2,
        'n_s2': len(idx_s2),
        'n_rect2': len(idx_rect2_list) // 2,
        'psnr_thr': psnr_thr,
    }


# ==================== 自适应解码 ====================

def decode_image_adaptive(enc, codebooks):
    HP, WP = enc['HP'], enc['WP']
    H, W = enc['H'], enc['W']
    H8, W8 = enc['H8'], enc['W8']
    sub8to4 = enc['sub8to4']
    idx_s8 = enc['idx_s8']
    idx_s4 = enc['idx_s4']
    idx_rect4 = enc.get('idx_rect4', np.array([], dtype=np.int32))
    idx_s2 = enc['idx_s2']
    idx_rect2 = enc.get('idx_rect2', np.array([], dtype=np.int32))

    is_v2 = enc.get('version', 1) >= 2

    cb88 = codebooks[(8, 8)][0]
    cb44 = codebooks[(4, 4)][0]
    cb22 = codebooks[(2, 2)][0]
    cb_2x4 = codebooks.get((2, 4), (None,))[0]
    cb_4x2 = codebooks.get((4, 2), (None,))[0]
    cb_1x2 = codebooks.get((1, 2), (None,))[0]
    cb_2x1 = codebooks.get((2, 1), (None,))[0]

    if is_v2:
        sub4_type = enc['sub4_type']
        sub2_type = enc['sub2_type']
    else:
        sub4to2 = enc.get('sub4to2', np.zeros((H8, W8, 2, 2), dtype=bool))
        sub4_type = np.where(sub4to2, 3, 0).astype(np.int8)
        sub2_type = np.zeros((H8, W8, 2, 2, 2, 2), dtype=np.int8)

    recon = np.zeros((HP, WP, 3), dtype=np.float32)
    p8, p4, pr4, p2, pr2 = 0, 0, 0, 0, 0

    for r8 in range(H8):
        for c8 in range(W8):
            if not sub8to4[r8, c8]:
                recon[r8 * 8:r8 * 8 + 8, c8 * 8:c8 * 8 + 8] = \
                    cb88[idx_s8[p8]].reshape(8, 8, 3).clip(0, 255)
                p8 += 1
            else:
                for sr in range(2):
                    for sc in range(2):
                        r4 = r8 * 8 + sr * 4
                        c4 = c8 * 8 + sc * 4
                        bt = sub4_type[r8, c8, sr, sc]

                        if bt == 0:
                            recon[r4:r4 + 4, c4:c4 + 4] = \
                                cb44[idx_s4[p4]].reshape(4, 4, 3).clip(0, 255)
                            p4 += 1
                        elif bt == 1:
                            recon[r4:r4 + 2, c4:c4 + 4] = \
                                cb_2x4[idx_rect4[pr4]].reshape(2, 4, 3).clip(0, 255)
                            pr4 += 1
                            recon[r4 + 2:r4 + 4, c4:c4 + 4] = \
                                cb_2x4[idx_rect4[pr4]].reshape(2, 4, 3).clip(0, 255)
                            pr4 += 1
                        elif bt == 2:
                            recon[r4:r4 + 4, c4:c4 + 2] = \
                                cb_4x2[idx_rect4[pr4]].reshape(4, 2, 3).clip(0, 255)
                            pr4 += 1
                            recon[r4:r4 + 4, c4 + 2:c4 + 4] = \
                                cb_4x2[idx_rect4[pr4]].reshape(4, 2, 3).clip(0, 255)
                            pr4 += 1
                        elif bt == 3:
                            for ssr in range(2):
                                for ssc in range(2):
                                    r2 = r4 + ssr * 2
                                    c2 = c4 + ssc * 2
                                    st = sub2_type[r8, c8, sr, sc, ssr, ssc]
                                    if st == 0:
                                        recon[r2:r2 + 2, c2:c2 + 2] = \
                                            cb22[idx_s2[p2]].reshape(2, 2, 3).clip(0, 255)
                                        p2 += 1
                                    elif st == 1:
                                        recon[r2:r2 + 1, c2:c2 + 2] = \
                                            cb_1x2[idx_rect2[pr2]].reshape(1, 2, 3).clip(0, 255)
                                        pr2 += 1
                                        recon[r2 + 1:r2 + 2, c2:c2 + 2] = \
                                            cb_1x2[idx_rect2[pr2]].reshape(1, 2, 3).clip(0, 255)
                                        pr2 += 1
                                    elif st == 2:
                                        recon[r2:r2 + 2, c2:c2 + 1] = \
                                            cb_2x1[idx_rect2[pr2]].reshape(2, 1, 3).clip(0, 255)
                                        pr2 += 1
                                        recon[r2:r2 + 2, c2 + 1:c2 + 2] = \
                                            cb_2x1[idx_rect2[pr2]].reshape(2, 1, 3).clip(0, 255)
                                        pr2 += 1

    return recon[:H, :W].astype(np.uint8)


# ==================== 统一词汇表 Token 编解码 ====================
#
# 所有码本索引连续累加，token 值本身标识码本类型，无需类型码。
#
# 例 K=1024:
#   8×8:       0    - 1023
#   SUB8:      1024          (细分信号)
#   4×4:       1025 - 2048
#   2×4:       2049 - 3072   (出现一次后，自动再读一个)
#   4×2:       3073 - 4096   (同上)
#   SUB4:      4097          (细分信号)
#   2×2:       4098 - 5121
#   1×2:       5122 - 6145   (出现一次后，自动再读一个)
#   2×1:       6146 - 7169   (同上)

def compute_vocab_offsets(codebooks):
    K8 = codebooks[(8, 8)][1]
    K4 = codebooks[(4, 4)][1]
    K2 = codebooks[(2, 2)][1]
    K_2x4 = codebooks.get((2, 4), (None, 0))[1]
    K_4x2 = codebooks.get((4, 2), (None, 0))[1]
    K_1x2 = codebooks.get((1, 2), (None, 0))[1]
    K_2x1 = codebooks.get((2, 1), (None, 0))[1]

    b = 0
    r_8x8 = (b, b + K8); b += K8
    r_sub8 = b; b += 1
    r_4x4 = (b, b + K4); b += K4
    r_2x4 = (b, b + K_2x4); b += K_2x4
    r_4x2 = (b, b + K_4x2); b += K_4x2
    r_sub4 = b; b += 1
    r_2x2 = (b, b + K2); b += K2
    r_1x2 = (b, b + K_1x2); b += K_1x2
    r_2x1 = (b, b + K_2x1); b += K_2x1
    return {
        '8x8': r_8x8, 'sub8': r_sub8,
        '4x4': r_4x4, '2x4': r_2x4, '4x2': r_4x2,
        'sub4': r_sub4,
        '2x2': r_2x2, '1x2': r_1x2, '2x1': r_2x1,
        'total': b,
        'K': [K8, K4, K_2x4, K_4x2, K2, K_1x2, K_2x1],
    }


def enc_to_tokens(enc, off):
    tokens = []
    H8, W8 = enc['H8'], enc['W8']
    is_v2 = enc.get('version', 1) >= 2

    sub8to4 = enc['sub8to4']
    if is_v2:
        sub4_type = enc['sub4_type']
        sub2_type = enc['sub2_type']
    else:
        s42 = enc.get('sub4to2', np.zeros((H8, W8, 2, 2), dtype=bool))
        sub4_type = np.where(s42, 3, 0).astype(np.int8)
        sub2_type = np.zeros((H8, W8, 2, 2, 2, 2), dtype=np.int8)

    i8 = enc['idx_s8']
    i4 = enc['idx_s4']
    ir4 = enc.get('idx_rect4', np.array([], dtype=np.int32))
    i2 = enc['idx_s2']
    ir2 = enc.get('idx_rect2', np.array([], dtype=np.int32))

    p8 = p4 = pr4 = p2 = pr2 = 0
    o = off

    for r8 in range(H8):
        for c8 in range(W8):
            if not sub8to4[r8, c8]:
                tokens.append(int(i8[p8])); p8 += 1
            else:
                tokens.append(o['sub8'])
                for sr in range(2):
                    for sc in range(2):
                        bt = int(sub4_type[r8, c8, sr, sc])
                        if bt == 0:
                            tokens.append(o['4x4'][0] + int(i4[p4])); p4 += 1
                        elif bt == 1:
                            tokens.append(o['2x4'][0] + int(ir4[pr4])); pr4 += 1
                            tokens.append(o['2x4'][0] + int(ir4[pr4])); pr4 += 1
                        elif bt == 2:
                            tokens.append(o['4x2'][0] + int(ir4[pr4])); pr4 += 1
                            tokens.append(o['4x2'][0] + int(ir4[pr4])); pr4 += 1
                        elif bt == 3:
                            tokens.append(o['sub4'])
                            for ssr in range(2):
                                for ssc in range(2):
                                    st = int(sub2_type[r8, c8, sr, sc, ssr, ssc])
                                    if st == 0:
                                        tokens.append(o['2x2'][0] + int(i2[p2])); p2 += 1
                                    elif st == 1:
                                        tokens.append(o['1x2'][0] + int(ir2[pr2])); pr2 += 1
                                        tokens.append(o['1x2'][0] + int(ir2[pr2])); pr2 += 1
                                    elif st == 2:
                                        tokens.append(o['2x1'][0] + int(ir2[pr2])); pr2 += 1
                                        tokens.append(o['2x1'][0] + int(ir2[pr2])); pr2 += 1
    return tokens


def decode_from_tokens(tokens, H, W, codebooks, off):
    cb88 = codebooks[(8, 8)][0]
    cb44 = codebooks[(4, 4)][0]
    cb22 = codebooks[(2, 2)][0]
    cb_2x4 = codebooks.get((2, 4), (None,))[0]
    cb_4x2 = codebooks.get((4, 2), (None,))[0]
    cb_1x2 = codebooks.get((1, 2), (None,))[0]
    cb_2x1 = codebooks.get((2, 1), (None,))[0]

    o = off
    r88 = o['8x8']; r44 = o['4x4']; r24 = o['2x4']; r42 = o['4x2']; r22 = o['2x2']
    r12 = o['1x2']; r21 = o['2x1']

    ph = (8 - H % 8) % 8; pw = (8 - W % 8) % 8
    HP, WP = H + ph, W + pw
    H8, W8 = HP // 8, WP // 8
    recon = np.zeros((HP, WP, 3), dtype=np.float32)
    pos = 0

    def rd():
        nonlocal pos; v = tokens[pos]; pos += 1; return v

    def in_range(v, r):
        return r[0] <= v < r[1]

    for r8 in range(H8):
        for c8 in range(W8):
            t = rd()
            if in_range(t, r88):
                recon[r8*8:r8*8+8, c8*8:c8*8+8] = \
                    cb88[t - r88[0]].reshape(8, 8, 3).clip(0, 255)
            elif t == o['sub8']:
                for sr in range(2):
                    for sc in range(2):
                        r4 = r8*8 + sr*4; c4 = c8*8 + sc*4
                        t = rd()
                        if in_range(t, r44):
                            recon[r4:r4+4, c4:c4+4] = \
                                cb44[t - r44[0]].reshape(4, 4, 3).clip(0, 255)
                        elif in_range(t, r24):
                            t2 = rd()
                            recon[r4:r4+2, c4:c4+4] = \
                                cb_2x4[t - r24[0]].reshape(2, 4, 3).clip(0, 255)
                            recon[r4+2:r4+4, c4:c4+4] = \
                                cb_2x4[t2 - r24[0]].reshape(2, 4, 3).clip(0, 255)
                        elif in_range(t, r42):
                            t2 = rd()
                            recon[r4:r4+4, c4:c4+2] = \
                                cb_4x2[t - r42[0]].reshape(4, 2, 3).clip(0, 255)
                            recon[r4:r4+4, c4+2:c4+4] = \
                                cb_4x2[t2 - r42[0]].reshape(4, 2, 3).clip(0, 255)
                        elif t == o['sub4']:
                            for ssr in range(2):
                                for ssc in range(2):
                                    r2 = r4 + ssr*2; c2 = c4 + ssc*2
                                    t = rd()
                                    if in_range(t, r22):
                                        recon[r2:r2+2, c2:c2+2] = \
                                            cb22[t - r22[0]].reshape(2, 2, 3).clip(0, 255)
                                    elif in_range(t, r12):
                                        t2 = rd()
                                        recon[r2:r2+1, c2:c2+2] = \
                                            cb_1x2[t - r12[0]].reshape(1, 2, 3).clip(0, 255)
                                        recon[r2+1:r2+2, c2:c2+2] = \
                                            cb_1x2[t2 - r12[0]].reshape(1, 2, 3).clip(0, 255)
                                    elif in_range(t, r21):
                                        t2 = rd()
                                        recon[r2:r2+2, c2:c2+1] = \
                                            cb_2x1[t - r21[0]].reshape(2, 1, 3).clip(0, 255)
                                        recon[r2:r2+2, c2+1:c2+2] = \
                                            cb_2x1[t2 - r21[0]].reshape(2, 1, 3).clip(0, 255)

    return recon[:H, :W].astype(np.uint8)


def compute_bpp_tokens(tokens, H, W, off):
    bits = math.ceil(math.log2(max(off['total'], 2)))
    total_bits = len(tokens) * bits
    return total_bits / (H * W), total_bits


# ==================== 平滑后处理 ====================

def smooth_deblocked(recon, enc=None, strength=0.6):
    """
    块边界感知去块效应滤波：
    仅在相邻块的边界像素上做加权平均，保留块内细节。
    strength: 0~1, 越大平滑越强
    """
    h, w = recon.shape[:2]
    out = recon.astype(np.float32).copy()
    radius = max(1, int(strength * 2))

    def _smooth_boundary(arr, axis, pos):
        if axis == 0:
            patch = arr[max(0, pos - radius):pos + radius + 1, :, :]
            if patch.shape[0] < 3:
                return arr
            kernel_1d = np.array([0.05, 0.15, 0.6, 0.15, 0.05], dtype=np.float32)
            if len(kernel_1d) > patch.shape[0]:
                k = patch.shape[0]
                kernel_1d = np.ones(k, dtype=np.float32) / k
            blended = np.zeros_like(patch)
            for i in range(patch.shape[0]):
                w_sum = 0.0
                val = np.zeros_like(patch[0], dtype=np.float32)
                for j in range(len(kernel_1d)):
                    idx = i - len(kernel_1d) // 2 + j
                    if 0 <= idx < patch.shape[0]:
                        val += patch[idx] * kernel_1d[j]
                        w_sum += kernel_1d[j]
                blended[i] = val / w_sum
            arr[max(0, pos - radius):pos + radius + 1, :, :] = \
                arr[max(0, pos - radius):pos + radius + 1, :, :].astype(np.float32) * (1 - strength) + \
                blended * strength
        else:
            patch = arr[:, max(0, pos - radius):pos + radius + 1, :]
            if patch.shape[1] < 3:
                return arr
            kernel_1d = np.array([0.05, 0.15, 0.6, 0.15, 0.05], dtype=np.float32)
            if len(kernel_1d) > patch.shape[1]:
                k = patch.shape[1]
                kernel_1d = np.ones(k, dtype=np.float32) / k
            blended = np.zeros_like(patch)
            for i in range(patch.shape[1]):
                w_sum = 0.0
                val = np.zeros_like(patch[:, 0, :], dtype=np.float32)
                for j in range(len(kernel_1d)):
                    idx = i - len(kernel_1d) // 2 + j
                    if 0 <= idx < patch.shape[1]:
                        val += patch[:, idx, :] * kernel_1d[j]
                        w_sum += kernel_1d[j]
                blended[:, i, :] = val / w_sum
            arr[:, max(0, pos - radius):pos + radius + 1, :] = \
                arr[:, max(0, pos - radius):pos + radius + 1, :].astype(np.float32) * (1 - strength) + \
                blended * strength
        return arr

    sizes = [8, 4, 2]
    for bs in sizes:
        for r in range(bs, h, bs):
            out = _smooth_boundary(out, 0, r)
        for c in range(bs, w, bs):
            out = _smooth_boundary(out, 1, c)

    return np.clip(out, 0, 255).astype(np.uint8)


def smooth_bilateral(recon, d=9, sigma_color=75, sigma_space=75):
    """双边滤波：保边平滑，消除块效应和量化噪声"""
    img = cv2.bilateralFilter(recon, d, sigma_color, sigma_space)
    return img


def smooth_gaussian(recon, sigma=1.0):
    """高斯模糊：简单平滑"""
    ksize = int(sigma * 6) | 1
    if ksize < 3:
        ksize = 3
    return cv2.GaussianBlur(recon, (ksize, ksize), sigma)


def smooth_median(recon, ksize=3):
    """中值滤波：去除椒盐噪声"""
    return cv2.medianBlur(recon, ksize)


def smooth_nlm(recon, h=10, template_window=7, search_window=21):
    """非局部均值去噪：效果好但速度较慢"""
    return cv2.fastNlMeansDenoisingColored(recon, None, h, h,
                                           template_window, search_window)


def smooth_edge_preserving(recon, flags=1, sigma_s=60, sigma_r=0.4):
    """
    边缘保留滤波 (cv2.edgePreservingFilter)
    flags=1 RECURS_FILTER:  递归滤波，速度快，效果自然
    flags=2 NORMCONV_FILTER: 归一化卷积，质量更高但较慢
    sigma_s: 空间范围 0~200
    sigma_r: 色彩范围 0~1
    """
    return cv2.edgePreservingFilter(recon, flags=flags,
                                    sigma_s=sigma_s, sigma_r=sigma_r)


def smooth_detail_enhance(recon, sigma_s=10, sigma_r=0.15):
    """
    细节增强滤波 (cv2.detailEnhance)
    在平滑的同时增强边缘细节，对 VQ 块效应有奇效
    """
    return cv2.detailEnhance(recon, sigma_s=sigma_s, sigma_r=sigma_r)


def smooth_guided(recon, radius=8, eps=0.01):
    """
    导向滤波 (Guided Filter) — numpy 实现
    使用自身作为引导图像，本质是边缘感知的局部线性滤波。
    radius: 窗口半径（像素），建议 4~16
    eps: 正则化参数，越小保留越多细节，越大越平滑
    """
    def _box_filter(img, r):
        ksize = 2 * r + 1
        return cv2.blur(img, (ksize, ksize), borderType=cv2.BORDER_REFLECT)

    def _guided_filter_single(p, I, r, eps):
        mean_I = _box_filter(I, r)
        mean_p = _box_filter(p, r)
        corr_Ip = _box_filter(I * p, r)
        var_I = _box_filter(I * I, r) - mean_I * mean_I

        a = (corr_Ip - mean_I * mean_p) / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = _box_filter(a, r)
        mean_b = _box_filter(b, r)

        q = mean_a * I + mean_b
        return q

    recon_f = recon.astype(np.float32) / 255.0
    out = np.zeros_like(recon_f)
    for c in range(3):
        out[:, :, c] = _guided_filter_single(
            recon_f[:, :, c], recon_f[:, :, c], radius, eps)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def smooth_multibilateral(recon, passes=3, d=9, sigma_color=75, sigma_space=75):
    """
    多次双边滤波：逐次迭代平滑，每次迭代保留边缘的同时
    逐步消除量化噪声和块边界，效果远超单次 bilateral。
    """
    result = recon.copy()
    for i in range(passes):
        result = cv2.bilateralFilter(result, d, sigma_color, sigma_space)
    return result


def smooth_image(recon, method='deblock', enc=None, **kwargs):
    """
    对解码图像进行平滑后处理。
    method 可选:
      deblock       — 块边界感知去块效应
      bilateral     — 单次双边滤波
      multibilat    — 多次迭代双边滤波（推荐）
      gaussian      — 高斯模糊
      median        — 中值滤波
      nlm           — 非局部均值去噪
      edge          — cv2 边缘保留滤波 (RECURS_FILTER)
      edge_norm     — cv2 边缘保留滤波 (NORMCONV_FILTER, 更高质量)
      detail        — cv2 细节增强
      guided        — 导向滤波 (Guided Filter)
      combo         — deblock + bilateral
      combo2        — guided + detail_enhance (推荐组合)
      combo3        — multibilat + edge_preserving (最强组合)
    """
    if method == 'none' or method is None:
        return recon

    if isinstance(recon, np.ndarray) and recon.dtype != np.uint8:
        recon = np.clip(recon, 0, 255).astype(np.uint8)

    print(f"    [平滑] 方法={method}", end='')
    t0 = time.time()

    if method == 'deblock':
        strength = kwargs.get('strength', 0.6)
        result = smooth_deblocked(recon, enc, strength)
        print(f"  strength={strength}", end='')
    elif method == 'bilateral':
        d = kwargs.get('d', 9)
        sc = kwargs.get('sigma_color', 75)
        ss = kwargs.get('sigma_space', 75)
        result = smooth_bilateral(recon, d, sc, ss)
        print(f"  d={d} sigma_c={sc} sigma_s={ss}", end='')
    elif method == 'multibilat':
        passes = kwargs.get('passes', 3)
        result = smooth_multibilateral(recon, passes=passes)
        print(f"  passes={passes}", end='')
    elif method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        result = smooth_gaussian(recon, sigma)
        print(f"  sigma={sigma}", end='')
    elif method == 'median':
        ksize = kwargs.get('ksize', 3)
        result = smooth_median(recon, ksize)
        print(f"  ksize={ksize}", end='')
    elif method == 'nlm':
        h_val = kwargs.get('h', 10)
        result = smooth_nlm(recon, h_val)
        print(f"  h={h_val}", end='')
    elif method == 'edge':
        ss = kwargs.get('sigma_s', 60)
        sr = kwargs.get('sigma_r', 0.4)
        result = smooth_edge_preserving(recon, flags=1, sigma_s=ss, sigma_r=sr)
        print(f"  RECURS sigma_s={ss} sigma_r={sr}", end='')
    elif method == 'edge_norm':
        ss = kwargs.get('sigma_s', 60)
        sr = kwargs.get('sigma_r', 0.4)
        result = smooth_edge_preserving(recon, flags=2, sigma_s=ss, sigma_r=sr)
        print(f"  NORMCONV sigma_s={ss} sigma_r={sr}", end='')
    elif method == 'detail':
        ss = kwargs.get('sigma_s', 10)
        sr = kwargs.get('sigma_r', 0.15)
        result = smooth_detail_enhance(recon, sigma_s=ss, sigma_r=sr)
        print(f"  sigma_s={ss} sigma_r={sr}", end='')
    elif method == 'guided':
        radius = kwargs.get('radius', 8)
        eps = kwargs.get('eps', 0.01)
        result = smooth_guided(recon, radius=radius, eps=eps)
        print(f"  radius={radius} eps={eps}", end='')
    elif method == 'combo':
        result = smooth_deblocked(recon, enc, strength=0.4)
        result = smooth_bilateral(result, d=7, sigma_color=50, sigma_space=50)
        print(f"  deblock(0.4)+bilateral(7,50,50)", end='')
    elif method == 'combo2':
        result = smooth_guided(recon, radius=8, eps=0.02)
        result = smooth_detail_enhance(result, sigma_s=10, sigma_r=0.12)
        print(f"  guided(r=8,eps=0.02)+detail(10,0.12)", end='')
    elif method == 'combo3':
        result = smooth_multibilateral(recon, passes=3, d=9,
                                       sigma_color=50, sigma_space=50)
        result = smooth_edge_preserving(result, flags=1,
                                        sigma_s=50, sigma_r=0.3)
        print(f"  multibilat(3)+edge(RECURS,s=50,r=0.3)", end='')
    else:
        print(f"  [警告] 未知方法 '{method}'，跳过")
        return recon

    elapsed = time.time() - t0
    print(f"  耗时={elapsed:.2f}s")
    return result


# ==================== 保存 / 加载 ====================

def save_tokens(tokens, H, W, off, path):
    np.savez_compressed(path,
                        tokens=np.array(tokens, dtype=np.int32),
                        H=H, W=W,
                        K_values=np.array(off['K'], dtype=np.int32))
    size_kb = os.path.getsize(path) / 1024
    print(f"    编码已保存: {path} ({size_kb:.1f}KB, "
          f"{len(tokens)} tokens, vocab={off['total']})")


def load_tokens(path):
    d = np.load(path)
    if 'tokens' in d:
        return d['tokens'].tolist(), int(d['H']), int(d['W']), \
               d['K_values'].tolist() if 'K_values' in d else None
    enc = _load_enc_dict(d)
    return None, enc['H'], enc['W'], None, enc


def _load_enc_dict(d):
    ver = int(d.get('version', 1))
    enc = {
        'version': ver,
        'H': int(d['H']), 'W': int(d['W']),
        'HP': int(d['HP']), 'WP': int(d['WP']),
        'H8': int(d['H8']), 'W8': int(d['W8']),
        'K8': int(d['K8']), 'K4': int(d['K4']), 'K2': int(d['K2']),
        'sub8to4': d['sub8to4'],
        'idx_s8': d['idx_s8'],
        'idx_s4': d['idx_s4'],
        'idx_s2': d['idx_s2'],
    }
    if ver >= 2:
        enc['sub4_type'] = d['sub4_type']
        enc['sub2_type'] = d['sub2_type']
        enc['idx_rect4'] = d['idx_rect4']
        enc['idx_rect2'] = d['idx_rect2']
    else:
        s42 = d.get('sub4to2', np.zeros((enc['H8'], enc['W8'], 2, 2), dtype=bool))
        enc['sub4_type'] = np.where(s42, 3, 0).astype(np.int8)
        enc['sub2_type'] = np.zeros((enc['H8'], enc['W8'], 2, 2, 2, 2), dtype=np.int8)
        enc['idx_rect4'] = np.array([], dtype=np.int32)
        enc['idx_rect2'] = np.array([], dtype=np.int32)
    return enc


# ==================== 单图处理 ====================

def encode_single(image_path, codebooks, off, psnr_thr, output_dir=None,
                   save_enc_path=None, smooth_method='none', smooth_strength=0.6):
    basename = os.path.basename(image_path)
    print(f"\n[编码] {basename}  (阈值={psnr_thr}dB)")

    orig = np.array(Image.open(image_path).convert('RGB'))
    H, W = orig.shape[:2]
    print(f"    尺寸: {W}×{H}")

    t0 = time.time()
    enc = encode_image_adaptive(orig, codebooks, psnr_thr)
    tokens = enc_to_tokens(enc, off)
    recon = decode_from_tokens(tokens, H, W, codebooks, off)

    psnr_raw, mse = compute_psnr(orig, recon)
    if smooth_method != 'none':
        recon_smooth = smooth_image(recon, smooth_method, enc,
                                    strength=smooth_strength)
        psnr_smooth, _ = compute_psnr(orig, recon_smooth)
        print(f"    平滑: PSNR {psnr_raw:.2f} → {psnr_smooth:.2f}dB")
        recon = recon_smooth
        psnr = psnr_smooth
    else:
        psnr = psnr_raw

    elapsed = time.time() - t0
    bpp, total_bits = compute_bpp_tokens(tokens, H, W, off)

    print(f"    结果: PSNR={psnr:.2f}dB  BPP={bpp:.4f}  "
          f"tokens={len(tokens)}  vocab={off['total']}  "
          f"耗时={elapsed:.2f}s")

    out_dir = output_dir or WORK_DIR
    cmp_path = os.path.join(out_dir, f'compare_{basename}')
    Image.fromarray(np.hstack([orig, recon])).save(cmp_path)
    print(f"    对比图: {cmp_path}")

    if save_enc_path:
        save_tokens(tokens, H, W, off, save_enc_path)

    return {
        'image': basename, 'psnr': psnr, 'bpp': bpp,
        'total_bits': total_bits, 'n_tokens': len(tokens),
        'n_s8': enc['n_s8'], 'n_s4': enc['n_s4'],
        'n_rect4': enc.get('n_rect4', 0),
        'n_s2': enc['n_s2'], 'n_rect2': enc.get('n_rect2', 0),
        'time': elapsed,
    }


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description='多尺度自适应VQ编解码')
    parser.add_argument('--cb-dir', type=str, default=None,
                        help='码本目录（自动检测所有码本）')
    parser.add_argument('--psnr', type=float, default=26,
                        help='PSNR 阈值(dB)')
    parser.add_argument('image', nargs='?', default='img_1.png', help='输入图片')
    parser.add_argument('--all', action='store_true', help='处理目录下所有图片')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--save-enc', type=str, default="test_image.tokens.npz",
                        help='编码结果保存路径')
    parser.add_argument('--decode', type=str, default=None,
                        help='编码文件路径（解码模式）')
    parser.add_argument('--ref', type=str, default=None,
                        help='解码模式下的原图')
    parser.add_argument('--smooth', type=str, default='combo2',
                        choices=['none', 'deblock', 'bilateral', 'multibilat',
                                 'gaussian', 'median', 'nlm',
                                 'edge', 'edge_norm', 'detail', 'guided',
                                 'combo', 'combo2', 'combo3'],
                        help='解码后平滑方法 (默认: combo2)')
    parser.add_argument('--smooth-strength', type=float, default=0.6,
                        help='deblock 平滑强度 0~1')
    args = parser.parse_args()

    cb_paths = find_codebooks(args.cb_dir)
    for key in [(8, 8), (4, 4), (2, 2)]:
        if key not in cb_paths:
            print(f"[错误] 缺少 {key[0]}×{key[1]} 码本"); return

    codebooks = load_all_codebooks(cb_paths)
    off = compute_vocab_offsets(codebooks)

    print(f"[词汇表] total={off['total']}  "
          f"8×8[0,{off['8x8'][1]})  SUB8={off['sub8']}  "
          f"4×4[{off['4x4'][0]},{off['4x4'][1]})  "
          f"2×4[{off['2x4'][0]},{off['2x4'][1]})  "
          f"4×2[{off['4x2'][0]},{off['4x2'][1]})  "
          f"SUB4={off['sub4']}  "
          f"2×2[{off['2x2'][0]},{off['2x2'][1]})  "
          f"1×2[{off['1x2'][0]},{off['1x2'][1]})  "
          f"2×1[{off['2x1'][0]},{off['2x1'][1]})")

    # ---- 解码模式 ----
    if args.decode:
        result = load_tokens(args.decode)
        tokens, H, W = result[0], result[1], result[2]
        if tokens is None:
            enc = result[4]
            tokens = enc_to_tokens(enc, off)
            H, W = enc['H'], enc['W']
        print(f"[解码] {H}×{W}, {len(tokens)} tokens")
        recon = decode_from_tokens(tokens, H, W, codebooks, off)
        if args.smooth != 'none':
            recon = smooth_image(recon, args.smooth, None,
                                 strength=args.smooth_strength)
        ref_path = args.ref or args.image
        if os.path.exists(ref_path):
            orig = np.array(Image.open(ref_path).convert('RGB'))
            psnr, _ = compute_psnr(orig, recon)
            print(f"[解码] PSNR={psnr:.2f}dB")
            out_dir = args.output or WORK_DIR
            out_path = os.path.join(out_dir,
                                    f'decoded_{os.path.basename(ref_path)}')
            Image.fromarray(np.hstack([orig, recon])).save(out_path)
            print(f"    对比图: {out_path}")
        else:
            out_dir = args.output or WORK_DIR
            out_path = os.path.join(out_dir, 'decoded.png')
            Image.fromarray(recon).save(out_path)
            print(f"[解码] 已保存: {out_path}")
        return

    # ---- 编码模式 ----
    if args.all:
        image_files = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            paths = glob.glob(os.path.join(WORK_DIR, ext))
            paths = [p for p in paths
                     if 'codebook' not in p and 'compare_' not in os.path.basename(p)
                     and 'decoded_' not in os.path.basename(p)]
            image_files.extend(paths)
    else:
        image_files = [args.image]

    if not image_files:
        print("[错误] 没有找到要处理的图片"); return

    print("=" * 60)
    print(f"多尺度自适应VQ编码  阈值={args.psnr}dB  vocab={off['total']}")
    print("=" * 60)

    results = []
    for path in image_files:
        if not os.path.exists(path):
            path = os.path.join(WORK_DIR, path)
        if not os.path.exists(path):
            print(f"[跳过] {path}"); continue

        enc_path = args.save_enc
        if enc_path and len(image_files) > 1:
            base = os.path.splitext(os.path.basename(path))[0]
            enc_path = os.path.join(
                args.output or WORK_DIR, f'{base}.tokens.npz')

        r = encode_single(path, codebooks, off,
                          psnr_thr=args.psnr,
                          output_dir=args.output,
                          save_enc_path=enc_path,
                          smooth_method=args.smooth,
                          smooth_strength=args.smooth_strength)
        results.append(r)

    if len(results) > 1:
        print("\n" + "=" * 60)
        print("汇总结果"); print("=" * 60)
        hdr = (f"{'图片':<25} {'PSNR':>8} {'BPP':>8} {'tokens':>8} "
               f"{'8×8':>6} {'4×4':>6} {'rect4':>6} "
               f"{'2×2':>6} {'rect2':>6}")
        print(hdr); print("-" * len(hdr))
        for r in results:
            print(f"{r['image']:<25} {r['psnr']:>7.2f}dB {r['bpp']:>8.4f} "
                  f"{r['n_tokens']:>8} "
                  f"{r['n_s8']:>6} {r['n_s4']:>6} {r['n_rect4']:>6} "
                  f"{r['n_s2']:>6} {r['n_rect2']:>6}")
        avg = sum(r['psnr'] for r in results) / len(results)
        print("-" * len(hdr)); print(f"{'平均':<25} {avg:>7.2f}dB")


if __name__ == '__main__':
    main()
