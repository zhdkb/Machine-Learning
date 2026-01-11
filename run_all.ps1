# 一键运行脚本（PowerShell）
# 使用方法：在项目根目录运行此脚本
python -m pip install -r requirements.txt
python clustering_pipeline.py --data Tosches_turtle.h5ad --outdir results --epochs 50 --latent 32 --batch 128 --device cpu
