import os
import json
from datetime import datetime
import glob
import zipfile

ROOT = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')

# 找到最新的结果文件夹
subdirs = [d for d in glob.glob(os.path.join(RESULTS_DIR, '*')) if os.path.isdir(d)]
if not subdirs:
    print('No results found in results/. Run the pipeline first.')
    exit(1)
latest = max(subdirs, key=os.path.getmtime)
print('Latest results dir:', latest)

# 读取metrics
metrics_path = os.path.join(latest, 'metrics.json')
if not os.path.exists(metrics_path):
    print('metrics.json not found in', latest)
    exit(1)
with open(metrics_path, 'r', encoding='utf-8') as f:
    metrics = json.load(f)

# 生成Markdown报告
report_md = os.path.join(latest, '聚类报告.md')
with open(report_md, 'w', encoding='utf-8') as f:
    f.write('# 单细胞转录组聚类分析报告\n\n')
    f.write('**数据集**: TURTLE (Tosches et al., 2018)\n\n')
    f.write('**生成时间**: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
    f.write('## 一、方法概述\n')
    f.write('- 数据预处理: 基因和细胞过滤、每细胞归一化到1e4、log1p、选取2000个高变基因、z-score标准化\n')
    f.write('- 表示学习: 标准自编码器（512-256-latent-256-512）训练，损失为MSE\n')
    f.write('- 聚类: KMeans，聚类数=已知细胞类型数\n\n')
    f.write('## 二、评估指标\n')
    f.write(f"- NMI: {metrics.get('NMI', 'NA'): .4f}\\n")
    f.write(f"- ARI: {metrics.get('ARI', 'NA'): .4f}\\n")
    f.write(f"- ACC: {metrics.get('ACC', 'NA'): .4f}\\n\n")
    f.write('## 三、可视化\n')
    if os.path.exists(os.path.join(latest, 'umap_results.png')):
        f.write('### UMAP 与聚类结果\n')
        f.write('![](umap_results.png)\n\n')
    if os.path.exists(os.path.join(latest, 'confusion_heatmap.png')):
        f.write('### 聚类与真实细胞类型对应热图\n')
        f.write('![](confusion_heatmap.png)\n\n')
    f.write('## 四、结论与改进建议\n')
    f.write('- 当前方法能较好地回收细胞类型结构，但仍有改进空间（替换VAE、图聚类、整合多数据集等）。\n')

print('Report written to', report_md)

# 打包为 zip
zip_path = os.path.join(RESULTS_DIR, os.path.basename(latest) + '.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(latest):
        for file in files:
            full = os.path.join(root, file)
            arcname = os.path.relpath(full, start=latest)
            zf.write(full, arcname)
print('Packaged results to', zip_path)
