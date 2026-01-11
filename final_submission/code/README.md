项目：TURTLE 单细胞转录组聚类分析

说明
- 本项目实现了数据预处理、基于标准自编码器的表示学习、基于潜在表示的聚类、评估（NMI/ARI/ACC）以及可视化与结果保存。

快速运行（推荐在虚拟环境中）：

Windows PowerShell 示例：
```powershell
python -m pip install -r requirements.txt
python clustering_pipeline.py --data Tosches_turtle.h5ad --outdir results --epochs 50 --latent 32 --batch 128 --device cpu
```

输出文件（results/<timestamp>/）
- `umap_results.png`：UMAP 可视化（聚类与真实类型）
- `confusion_heatmap.png`：聚类-真实类型对应热图
- `metrics.json`：NMI/ARI/ACC 值
- `pred_labels.csv`：聚类标签

说明与扩展
- 若要改用 GPU，请在 Windows 上安装对应版本的 `torch` 并传入 `--device cuda`。
- 可在 `clustering_pipeline.py` 中调整自编码器结构、超参或替换成变分自编码器 / 去噪自编码器以进行改进。

评分要求对照
- 已实现：数据预处理、标准自编码器、聚类与评估、可视化、可复现运行脚本。
- 可扩展：多数据集验证、更多消融实验与改进模型（可按评分标准添加）。
