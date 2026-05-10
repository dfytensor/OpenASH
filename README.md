# Multi-Scale Adaptive VQ (MSA-VQ)

https://img.shields.io/badge/License-MIT-yellow.svg](https://opensource.org/licenses/MIT)
https://img.shields.io/badge/python-3.8+-blue.svg](https://www.python.org/downloads/release/python-380/)

**MSA-VQ** 是一套基于 **多尺度自适应矢量量化（Vector Quantization）** 的通用编解码框架。不同于传统的固定尺寸 VQ，本项目引入了**非对称块（如 2×4, 4×2）**和**五阶段自适应细分策略**，在保证极高重建质量（PSNR）的同时，实现了接近信息论极限的压缩效率。

本项目旨在提供一个**高效、可解释、面向科学计算**的离散表示底座，未来将无缝扩展至视频与音频领域，探索物理世界信号的统一建模。

---

## 🚀 核心特性

*   **🌍 多尺度自适应编码**：支持从 8×8 到 1×2 的 7 种不同尺度的码本，根据局部复杂度动态调整表示粒度（Pass 1 到 Pass 5）。
*   **⚡ 极致性能**：采用 NumPy 优化的批量最近邻搜索，支持 CPU/GPU 加速（需配置 CuPy），在低比特率下仍保持高保真度。
*   **🔬 科学导向的设计**：
    *   **零幻觉**：确定性编解码，无随机性，适合科学仿真与因果推断。
    *   **Token 化友好**：输出为统一词汇表的离散 Token，可直接接入 LLM/RWKV 等语言模型进行序列建模。
*   **🎨 丰富的后处理**：内置多种去块效应与边缘保留滤波算法（双边滤波、导向滤波、细节增强等），提升视觉质量。

---

## 📂 项目结构

```text
.
├── encode.py       # 核心编码器：将图像转换为多尺度 Token 序列
├── train.py        # 码本训练器：使用 K-Means 训练多尺度码本
└── codebooks/      # 预训练码本存放目录 (需自行训练或下载)
```

---

## 🛠️ 安装与环境

```bash
# 克隆仓库
git clone https://github.com/your_username/MSA-VQ.git
cd MSA-VQ

# 安装依赖
pip install -r requirements.txt
# 主要依赖：numpy, opencv-python, pillow, scikit-learn
```

---

## 🏃 快速开始

### 1. 训练码本
首先需要训练多尺度码本。默认使用 Mini-ImageNet 或本地图片。

```bash
# 训练所有尺度 (2x2, 4x4, 8x8, 2x4, 4x2, 1x2, 2x1)
python train.py --k 16384 --images 5000 --blocks 50

# 仅训练特定尺度 (例如只训练 8x8 和 4x4)
python train.py --scales 8 4 --k 8192
```

### 2. 编码与解码
使用训练好的码本对单张图片进行编码。

```bash
# 编码图片 (PSNR 阈值设为 30dB)
python encode.py img_1.png --psnr 30 --smooth combo2

# 解码 .tokens.npz 文件
python encode.py --decode test_image.tokens.npz --ref img_1.png
```

---

## 🔮 未来规划：通往通用物理表示

MSA-VQ 不仅仅是一个图像压缩工具，它是我们构建**统一物理世界模型**的第一步。我们坚信，物理世界的信号（视觉、听觉、触觉）在离散空间中具有统一的几何结构。

| 阶段 | 目标 | 状态 |
| :--- | :--- | :--- |
| **Phase 1** | **静态视觉** (Image) | ✅ **Current** |
| **Phase 2** | **动态视频** (Video) | 🚧 **In Progress** |
| **Phase 3** | **音频信号** (Audio) | 🚀 **Next** |
| **Phase 4** | **多模态统一** (Unified Tokenizer) | 🎯 **Target** |

### 为什么这很重要？
目前的深度学习模型（如 Diffusion 或 LLM）通常是“黑盒”且“数据饥渴”的。MSA-VQ 旨在通过**多尺度离散化**，捕捉信号的本质结构：
1.  **视频方向**：将时间维度纳入 VQ，实现长时序的物理过程模拟（如流体动力学），而非仅仅是帧预测。
2.  **音频方向**：利用非对称块（时域×频域）建模声波，探索听觉感知的离散编码原理。
3.  **科学发现**：结合轻量级 OpenASH 架构，在极低算力下（单张消费级显卡）复现复杂的物理规律，推动 **AI for Science** 的普惠化。

---

## 🤝 寻求合作与资源

我们正处于从 **Phase 1 (视觉)** 向 **Phase 2 & 3 (视频/音频)** 跨越的关键期。为了验证该框架在真实物理场景中的有效性，我们需要更多的计算资源与多模态数据集。

**我们正在寻找：**
*   拥有 **高性能计算集群 (A100/H100)** 的研究所或企业实验室。
*   在 **计算物理、神经科学、音频信号处理** 领域拥有独家数据的团队。

**我们能提供：**
*   一套经过验证的、**极高效率**的多尺度离散表示框架。
*   完整的开源生态与工程实现。
*   冲击 **Nature/Nature Machine Intelligence** 级别成果的创新潜力。

如果你认为这项研究有价值，请联系我们，让我们共同探索物理世界的离散本质。

---

## 📜 许可

本项目采用 **MIT License**。欢迎自由使用、修改和分发，但请注意保留版权声明。

**Note**: 如果您将本项目用于学术研究，请引用我们的工作（待定 DOI）。
