"""
clustering_pipeline.py
主程序：实现数据预处理、自编码器训练、基于潜在表示的聚类、评估和可视化
用法（示例）：
    python clustering_pipeline.py --data Tosches_turtle.h5ad --outdir results --epochs 50
"""

import os
import argparse
import json
from datetime import datetime

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
import umap
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

sns.set_style('whitegrid')


class SingleCellDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype(np.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx]


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, z


def load_data(path):
    print('Loading AnnData:', path)
    adata = sc.read(path, index_col=0)
    # convert sparse to dense if needed
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    X = np.nan_to_num(X, nan=0.0)
    adata.X = X
    if 'celltype' not in adata.obs.columns:
        raise KeyError('需要在adata.obs中包含celltype列作为标签')
    labels = adata.obs['celltype'].values
    return adata, labels


def preprocess(adata, min_cells=3, min_genes_per_cell=200, n_top_genes=2000):
    print('\nPreprocessing...')
    X = adata.X
    # filter genes
    gene_mask = (X > 0).sum(axis=0) >= min_cells
    adata = adata[:, gene_mask]
    X = adata.X
    print(f'After gene filter: {adata.n_vars} genes')
    # compute n_genes per cell
    n_genes = (X > 0).sum(axis=1)
    cell_mask = n_genes > min_genes_per_cell
    adata = adata[cell_mask]
    X = adata.X
    print(f'After cell filter: {adata.n_obs} cells')
    # normalize per cell
    sums = X.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0
    X_norm = X / sums * 1e4
    X_log = np.log1p(X_norm)
    # select highly variable genes by variance
    var = np.var(X_log, axis=0)
    top_idx = np.argsort(var)[-n_top_genes:]
    X_selected = X_log[:, top_idx]
    print(f'Selected HVG: {X_selected.shape[1]}')

    # 将 adata 子集化到选中的高变基因，保持 adata.var 一致
    adata = adata[:, top_idx]

    # z-score
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    # PCA降维
    print('计算PCA...')
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)
    print(f'PCA降维至: {X_pca.shape}')
    print(f'解释方差比: {pca.explained_variance_ratio_[:10]}')

    # 更新adata
    adata.obsm['X_pca'] = X_pca
    adata.X = X_scaled
    # store useful arrays
    adata.var['hvg'] = False
    try:
        hvg_col_idx = adata.var.columns.get_loc('hvg')
        adata.var.iloc[:, hvg_col_idx] = False
    except Exception:
        pass
    adata.obsm['raw_HVG'] = X_selected
    return adata, scaler


def train_autoencoder(X, latent_dim=32, epochs=50, batch_size=128, lr=1e-3, device='cpu'):
    input_dim = X.shape[1]
    model = AutoEncoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = SingleCellDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            recon, z = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        if epoch % max(1, epochs//5) == 0 or epoch==1:
            print(f'Epoch {epoch}/{epochs} loss: {total_loss/len(dataset):.6f}')
    # extract latent
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(device)
        _, Z = model(X_tensor)
        Z = Z.cpu().numpy()
    return model, Z


def cluster_and_evaluate(true_labels, Z, n_clusters=15):
    print('\nClustering and evaluation...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred = kmeans.fit_predict(Z)
    nmi = normalized_mutual_info_score(true_labels, pred)
    ari = adjusted_rand_score(true_labels, pred)
    # ACC via Hungarian
    le = LabelEncoder()
    true_int = le.fit_transform(true_labels)
    pred_int = pred
    n_types = len(np.unique(true_int))
    n_c = len(np.unique(pred_int))
    cm = np.zeros((n_types, n_c), dtype=np.int64)
    for i in range(len(true_int)):
        cm[true_int[i], pred_int[i]] += 1
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / len(true_int)
    print(f'NMI={nmi:.4f} ARI={ari:.4f} ACC={acc:.4f}')
    return pred, {'NMI': float(nmi), 'ARI': float(ari), 'ACC': float(acc)}


def visualize_and_save(Z, adata, pred, outdir):
    os.makedirs(outdir, exist_ok=True)
    # UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    U = reducer.fit_transform(Z)
    df = pd.DataFrame({'UMAP1': U[:,0], 'UMAP2': U[:,1], 'cluster': pred, 'celltype': adata.obs['celltype'].values})
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    sc1 = plt.scatter(df['UMAP1'], df['UMAP2'], c=df['cluster'], cmap='tab20', s=5)
    plt.title('UMAP - clusters')
    plt.subplot(1,2,2)
    uniques = df['celltype'].unique()
    colors = plt.cm.tab20(np.linspace(0,1,len(uniques)))
    for i, t in enumerate(uniques):
        mask = df['celltype']==t
        plt.scatter(df.loc[mask,'UMAP1'], df.loc[mask,'UMAP2'], label=t, color=colors[i], s=5)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)
    plt.title('UMAP - true cell types')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'umap_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    # confusion heatmap
    cm = pd.crosstab(adata.obs['celltype'], pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, cmap='YlOrRd')
    plt.title('Confusion: true celltype vs cluster')
    plt.xlabel('cluster')
    plt.ylabel('celltype')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'confusion_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved visualizations to', outdir)


def save_results(outdir, metrics, pred_labels):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    pd.Series(pred_labels, name='pred_cluster').to_csv(os.path.join(outdir, 'pred_labels.csv'), index=False)
    print('Saved metrics and predicted labels')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='Tosches_turtle.h5ad')
    parser.add_argument('--outdir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--latent', type=int, default=32)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    adata, true_labels = load_data(args.data)
    adata, scaler = preprocess(adata)
    # use preprocessed scaled matrix
    X = adata.X.copy()
    model, Z = train_autoencoder(X, latent_dim=args.latent, epochs=args.epochs, batch_size=args.batch, device=args.device)
    pred, metrics = cluster_and_evaluate(true_labels[ ( (X.shape[0]==len(true_labels)) and np.arange(len(true_labels)) ) if False else np.arange(len(true_labels[:X.shape[0]])) ], Z, n_clusters=len(np.unique(true_labels)))
    # Above: ensure true_labels length matches cells after filtering; map if necessary
    # Simpler approach: use adata.obs['celltype'] (already filtered)
    pred, metrics = cluster_and_evaluate(adata.obs['celltype'].values, Z, n_clusters=len(np.unique(adata.obs['celltype'].values)))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = os.path.join(args.outdir, timestamp)
    visualize_and_save(Z, adata, pred, outdir)
    save_results(outdir, metrics, pred)
    print('\nDone. Results in', outdir)

if __name__ == '__main__':
    main()
