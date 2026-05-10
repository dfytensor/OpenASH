#!/usr/bin/env python3
"""
train.py — 多尺度VQ码本训练
=============================

支持正方 (2×2, 4×4, 8×8) 和非正方 (2×4, 4×2, 1×2, 2×1) 尺度。

使用方式：
    python train.py                                  # 默认 K=16384, 训练全部尺度
    python train.py --k 32768                        # K=32768
    python train.py --scales 8 4 2                   # 只训练正方尺度
    python train.py --scales 2x4 4x2 1x2 2x1        # 只训练非正方尺度
    python train.py --k 16384 --scales 8 2x4 4x2     # 混合训练
    python train.py --help                           # 查看所有参数

输出：
    codebook_S2_K{K}.npz      — 2×2 正方码本
    codebook_S4_K{K}.npz      — 4×4 正方码本
    codebook_S8_K{K}.npz      — 8×8 正方码本
    codebook_S2x4_K{K}.npz    — 2×4 非正方码本
    codebook_S4x2_K{K}.npz    — 4×2 非正方码本
    codebook_S1x2_K{K}.npz    — 1×2 非正方码本
    codebook_S2x1_K{K}.npz    — 2×1 非正方码本
"""

import os, sys, time, math, gc, glob, argparse
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


WORK_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_scale_str(s):
    """解析尺度字符串: '8' -> (8,8), '2x4' -> (2,4)"""
    s = str(s)
    if 'x' in s:
        parts = s.split('x')
        return (int(parts[0]), int(parts[1]))
    return (int(s), int(s))


def scale_to_str(scale):
    """(8,8) -> 'S8', (2,4) -> 'S2x4'"""
    if scale[0] == scale[1]:
        return f'S{scale[0]}'
    return f'S{scale[0]}x{scale[1]}'


def scale_display(scale):
    """(8,8) -> '8×8', (2,4) -> '2×4'"""
    return f'{scale[0]}×{scale[1]}'


ALL_SCALES = [
    (1, 2), (2, 1), (2, 2), (2, 4), (4, 2), (4, 4), (8, 8),
]


def load_blocks_from_images(
    image_paths,
    block_h=8,
    block_w=8,
    blocks_per_image=500,
    resize_to=128,
):
    dim = block_h * block_w * 3
    blks = []
    total = len(image_paths)

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert('RGB')
            if resize_to:
                img = img.resize((resize_to, resize_to), Image.BILINEAR)
            pixels = np.array(img)
            H, W = pixels.shape[:2]

            if H < block_h or W < block_w:
                continue

            for _ in range(blocks_per_image):
                r = np.random.randint(0, H - block_h + 1)
                c = np.random.randint(0, W - block_w + 1)
                blk = pixels[r:r + block_h, c:c + block_w].copy()
                blks.append(blk)

        except Exception:
            pass

        if (i + 1) % 50 == 0:
            print(f"    加载进度: {i+1}/{total} 张图片，{len(blks)} 个块")
            sys.stdout.flush()

    blocks = np.array(blks, dtype=np.float32)
    del blks; gc.collect()

    print(f"    共提取 {len(blocks)} 个 {block_h}×{block_w} 块，shape={blocks.shape}")
    return blocks


def find_training_images(max_count=300):
    paths = []

    cache_root = os.path.join(os.path.expanduser('~'), '.cache', 'modelscope',
                              'hub', 'datasets', 'tany0699', 'mini_imagenet100')
    if os.path.isdir(cache_root):
        for dirpath, _, filenames in os.walk(cache_root):
            for fn in filenames:
                if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(dirpath, fn))
        if paths:
            print(f"[数据] Mini-ImageNet缓存: {len(paths)} 张")

    if not paths:
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            paths.extend(glob.glob(os.path.join(WORK_DIR, ext)))
        paths = [p for p in paths if 'v' not in os.path.basename(p)[:3]]
        print(f"[数据] 工作目录: {len(paths)} 张")

    np.random.shuffle(paths)
    paths = paths[:max_count]
    print(f"[数据] 使用 {len(paths)} 张进行训练")
    return paths


def train_codebook(
    blocks,
    block_h,
    block_w,
    K,
    batch_size=250000,
    max_iter=200,
    init='random',
    n_init=1,
):
    dim = block_h * block_w * 3
    X = blocks.reshape(len(blocks), -1).astype(np.float32)
    n_blocks = len(X)
    del blocks; gc.collect()

    if n_blocks < K:
        print(f"[警告] 训练块数({n_blocks}) < K({K})，将K降至{n_blocks}")
        K = n_blocks

    print(f"\n[训练] 尺度={block_h}×{block_w}, K={K}, 维度={dim}, "
          f"训练块数={n_blocks}, max_iter={max_iter}, init={init}")

    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=K,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=42,
        n_init=n_init,
        init=init,
        tol=1e-4,
        verbose=0,
    )
    km.fit(X)

    cb = km.cluster_centers_.astype(np.float32)
    elapsed = time.time() - t0

    print(f"[训练] 完成，inertia={km.inertia_:.0f}，耗时{elapsed:.1f}秒")
    del X, km; gc.collect()

    return cb


def save_codebook(cb, block_h, block_w, K, init, max_iter, output_path=None):
    if output_path is None:
        sname = scale_to_str((block_h, block_w))
        output_path = os.path.join(WORK_DIR, f'codebook_{sname}_K{K}.npz')

    np.savez_compressed(output_path,
                        codebook=cb,
                        block_h=block_h,
                        block_w=block_w,
                        block_size=block_h if block_h == block_w else block_h,
                        K=K,
                        init=init,
                        max_iter=max_iter)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"[保存] {output_path} ({size_mb:.1f}MB)")
    return output_path


def train_single_scale(scale, K, paths, blocks_per_image, max_iter, init, output=None):
    bh, bw = scale
    print(f"\n{'=' * 60}")
    print(f"训练 {bh}×{bw} 尺度码本 — K={K}")
    print(f"{'=' * 60}")

    print(f"\n[步骤1] 提取 {bh}×{bw} 块")
    blocks = load_blocks_from_images(
        paths,
        block_h=bh,
        block_w=bw,
        blocks_per_image=blocks_per_image,
        resize_to=128,
    )

    print(f"\n[步骤2] 训练码本")
    cb = train_codebook(blocks, block_h=bh, block_w=bw, K=K,
                        max_iter=max_iter, init=init)

    print(f"\n[步骤3] 保存码本")
    save_codebook(cb, block_h=bh, block_w=bw, K=K, init=init,
                  max_iter=max_iter, output_path=output)

    print(f"[完成] {bh}×{bw} 尺度码本已保存")
    return cb


def main():
    parser = argparse.ArgumentParser(description='多尺度VQ码本训练（支持非正方尺度）')
    parser.add_argument('--k', type=int, default=16384, help='聚类数K（码本词条数）')
    parser.add_argument('--scales', type=str, nargs='+',
                        default=['2', '4', '8', '2x4', '4x2', '1x2', '2x1'],
                        help='要训练的块尺度（如: 8 4 2 2x4 4x2 1x2 2x1）')
    parser.add_argument('--images', type=int, default=10000, help='使用的图片数量')
    parser.add_argument('--blocks', type=int, default=50, help='每张图每个尺度提取的块数')
    parser.add_argument('--iter', type=int, default=200, help='最大迭代次数')
    parser.add_argument('--init', type=str, default='random', choices=['random', 'k-means++'],
                        help='初始化方式：random快，k-means++质量高')
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件路径模板（仅单尺度时有效）')
    args = parser.parse_args()

    scales = [parse_scale_str(s) for s in args.scales]
    for s in scales:
        if s not in ALL_SCALES:
            print(f"[错误] 不支持的尺度: {scale_display(s)}，"
                  f"可选: {', '.join(scale_display(sc) for sc in ALL_SCALES)}")
            return

    print("=" * 60)
    print(f"多尺度VQ码本训练 — 尺度={[scale_display(s) for s in scales]}, K={args.k}, "
          f"{args.images}张图, {args.blocks}块/图/尺度")
    print("=" * 60)

    paths = find_training_images(max_count=args.images)

    for scale in scales:
        output = args.output if len(scales) == 1 and args.output else None
        train_single_scale(
            scale=scale,
            K=args.k,
            paths=paths,
            blocks_per_image=args.blocks,
            max_iter=args.iter,
            init=args.init,
            output=output,
        )

    print(f"\n{'=' * 60}")
    print(f"全部完成！已训练尺度: {[scale_display(s) for s in scales]}")
    print(f"码本文件: codebook_S{{尺度}}_K{args.k}.npz")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
