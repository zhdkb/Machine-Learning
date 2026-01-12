# 单细胞转录组聚类分析


---


## 📁 文件结构说明

```
submit/
├── README.md                          # 本文档
├── requirements.txt                   # Python依赖包列表
│
├── code/                              # 完整源代码（所有依赖代码）
│   ├── clustering_pipeline.py         # 基础任务主程序 (250行)
│   ├── innovative_clustering_methods.py # 7种创新算法库 (450行)
│   ├── multi_dataset_runner.py        # 多数据集批量运行 (380行)
│
├── data/                              # 所有输入数据文件
│   ├── Tosches_turtle.h5ad           # TURTLE数据集 (324MB，真实数据)
│   ├── Quake_Lung.h5ad               # Quake Lung (35MB，演示数据)
│   └── Quake_Diaphragm.h5ad          # Quake Diaphragm (15MB，演示数据)
│
├── results/                           # 所有实验结果
│   ├── all_results.json               # 完整原始结果数据
│   ├── method_comparison.csv          # 方法性能对比表格
│   └── figures/                       # 高质量可视化图表
│       ├── umap_results.png           # UMAP降维可视化
│       └── confusion_heatmap.png      # 混淆矩阵热图
│
└── scripts/                           # 运行脚本
    ├── run_all_comprehensive.ps1      # PowerShell一键运行脚本
```

---

## 🚀 快速开始指南

### 方案1: PowerShell一键运行（推荐）

```powershell
# 1. 进入项目目录
cd submit

# 2. 运行完整实验脚本
.\scripts\run_all_comprehensive.ps1

# 或运行简化版脚本（更快）
.\scripts\simple_run.ps1
```

### 方案2: 手动运行（逐步执行）

```bash
# 1. 进入项目目录并安装依赖
cd submit
python -m pip install -r requirements.txt

# 2. 基础任务 - 运行自编码器聚类（选择一个数据集）
python code/clustering_pipeline.py --data data/Tosches_turtle.h5ad --outdir results --epochs 50 --device cpu

# 3. 创新方法 - 运行7种算法的多数据集验证
python code/multi_dataset_runner.py
```

---

## 📊 核心运行参数说明

### clustering_pipeline.py 参数
- `--data`: 输入的H5AD数据文件路径
- `--outdir`: 结果输出目录（默认: results）
- `--epochs`: 自编码器训练轮次（默认: 50）
- `--latent`: 潜在表示维度（默认: 32）
- `--batch`: 批大小（默认: 128）
- `--device`: 计算设备，cpu或cuda（默认: cpu）
- `--seed`: 随机种子（默认: 42）

### 推荐运行配置
```bash
# GPU运行（加速训练）
python code/clustering_pipeline.py --data data/Tosches_turtle.h5ad --device cuda --epochs 100

# CPU运行（通用）
python code/clustering_pipeline.py --data data/Tosches_turtle.h5ad --device cpu --epochs 50

# 快速测试
python code/clustering_pipeline.py --data data/Tosches_turtle.h5ad --epochs 10 --batch 256
```

---

## 📈 性能评估结果汇总

### 最佳性能指标

| 数据集 | 最优方法 | NMI | ARI | ACC | 特点 |
|--------|---------|-----|-----|-----|------|
| **Tosches TURTLE** | Hierarchical | 0.6933 | 0.6574 | 0.6585 | 复杂真实数据 |
| **Quake-Lung** | Improved K-means | 0.9977 | 0.9978 | 0.9990 | 高质量演示数据 |
| **Quake-Diaphragm** | HDBSCAN | 0.9531 | 0.9374 | 0.9413 | 中等难度数据 |

### 方法性能排名

**TURTLE数据集** (最复杂，最具代表性)
1. Hierarchical Clustering - NMI: 0.6933 ⭐⭐⭐
2. Ensemble Voting - NMI: 0.6705 ⭐⭐⭐
3. Kernel K-means - NMI: 0.6512 ⭐⭐
4. HDBSCAN - NMI: 0.5892
5. GMM - NMI: 0.5112
6. DBSCAN - NMI: 0.4812
7. Improved K-means - NMI: 0.4621

详细分析见 `results/多数据集对比分析报告.md`

---

## 📝 关键文件说明

### 源代码文件

#### 1. clustering_pipeline.py（基础任务）
- **功能**: 完整的聚类分析流程
- **内容**: 
  - H5AD数据加载与预处理
  - 标准自编码器训练
  - K-means聚类
  - NMI/ARI/ACC计算
  - UMAP可视化与热图生成
- **输出**: results/<timestamp>/目录中包含umap_results.png、metrics.json等

#### 2. innovative_clustering_methods.py（创新算法库）
- **功能**: 7种聚类算法的完整实现
- **内容**:
  - ImprovedKMeans - 改进的K-means（Elkan + 多初始化）
  - HierarchicalClustering - 层级聚类（Ward链接）
  - GMMClustering - 高斯混合模型
  - KernelKMeans - 核K-means（RBF核）
  - DBSCANClustering - 密度聚类
  - HDBSCANClustering - 层级密度聚类
  - EnsembleVoting - 投票集成方法

#### 3. multi_dataset_runner.py（多数据集运行）
- **功能**: 在所有数据集上批量运行所有算法
- **内容**:
  - 并行处理3个数据集
  - 对每个数据集运行7种算法
  - 计算评估指标
  - 生成对比表格




### 结果文件

#### 1. all_results.json（原始数据）
```json
{
  "Tosches_turtle": {
    "Hierarchical": {"NMI": 0.6933, "ARI": 0.6574, "ACC": 0.6585},
    "Improved K-means": {"NMI": 0.4621, "ARI": 0.3876, "ACC": 0.4012},
    ...
  },
  "Quake_Lung": {...},
  "Quake_Diaphragm": {...}
}
```

#### 2. method_comparison.csv（对比表格）
易于导入Excel进行二次分析的表格形式，包含所有方法的完整性能指标。

#### 3. 多数据集对比分析报告.md
包含了每个数据集上运行的结果

---

## 🔧 依赖包与环境要求

### 环境要求
- **Python版本**: 3.7+ （推荐 3.9+）
- **操作系统**: Windows、Linux、macOS
- **内存**: 至少8GB（处理大型H5AD文件）
- **GPU**（可选）: CUDA 11.0+ 用于GPU加速

### 依赖包列表
```
scanpy            # 单细胞数据处理
numpy             # 数值计算
pandas            # 数据操作
matplotlib        # 绘图
seaborn           # 统计绘图
scikit-learn      # 机器学习算法
umap-learn        # UMAP降维
torch             # PyTorch（自编码器训练）
python-dateutil   # 日期时间处理
```

### 安装步骤

```bash
# 1. 进入项目目录
cd submit

# 2. 安装依赖（推荐虚拟环境）
python -m pip install -r requirements.txt

# 3. 可选：GPU支持（如需要）
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

