"""
innovative_clustering_methods.py
创新聚类方法实现（禁用自编码器和对比学习）
包含：改进K-means、层次聚类、GMM、HDBSCAN、核K-means等
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Warning: hdbscan not installed. Some methods will be unavailable.")


def compute_acc(true_labels, pred_labels):
    """计算ACC指标（通过匈牙利算法最优匹配）"""
    le_true = LabelEncoder()
    le_pred = LabelEncoder()
    true_int = le_true.fit_transform(true_labels)
    pred_int = le_pred.fit_transform(pred_labels)
    
    n_types = len(np.unique(true_int))
    n_clusters = len(np.unique(pred_int))
    
    cm = np.zeros((n_types, n_clusters), dtype=np.int64)
    for i in range(len(true_int)):
        cm[true_int[i], pred_int[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(-cm)
    acc = cm[row_ind, col_ind].sum() / len(true_int)
    return acc


def evaluate_clustering(true_labels, pred_labels):
    """计算NMI、ARI、ACC三个指标"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    acc = compute_acc(true_labels, pred_labels)
    return {'NMI': float(nmi), 'ARI': float(ari), 'ACC': float(acc)}


class ImprovedKMeans:
    """改进K-means：多初始化 + Elkan算法 + 收敛检测优化"""
    def __init__(self, n_clusters, n_init=20, max_iter=500, random_state=42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, 
                          random_state=random_state, algorithm='elkan')
    
    def fit_predict(self, X):
        return self.km.fit_predict(X)


class KernelKMeans:
    """核K-means：使用RBF核进行非线性聚类"""
    def __init__(self, n_clusters, gamma=1.0, n_init=10, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels = None
    
    def _rbf_kernel(self, X):
        """计算RBF核矩阵"""
        sq_dists = pdist(X, 'sqeuclidean')
        sq_dists = squareform(sq_dists)
        K = np.exp(-self.gamma * sq_dists)
        return K
    
    def fit_predict(self, X):
        """核K-means聚类"""
        K = self._rbf_kernel(X)
        np.random.seed(self.random_state)
        
        best_labels = None
        best_inertia = np.inf
        
        for init in range(self.n_init):
            # 随机初始化中心索引
            center_idx = np.random.choice(len(X), self.n_clusters, replace=False)
            labels = np.zeros(len(X), dtype=int)
            
            for it in range(self.max_iter):
                # 计算样本到中心的距离（在核空间中）
                K_cc = K[center_idx][:, center_idx]
                K_xc = K[:, center_idx]
                
                # 欧氏距离在核空间中的近似
                dists = np.diag(K)[:, np.newaxis] - 2*K_xc + np.diag(K_cc)
                dists = np.maximum(dists, 0)  # 数值稳定性
                new_labels = np.argmin(dists, axis=1)
                
                if np.all(new_labels == labels):
                    break
                labels = new_labels
                
                # 更新中心：选择每个簇中核矩阵行和最大的点
                for k in range(self.n_clusters):
                    mask = (labels == k)
                    if mask.sum() > 0:
                        center_idx[k] = np.argmax(K_xc[mask].sum(axis=1))
            
            # 计算惯性
            inertia = 0
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    inertia += dists[mask, k].sum()
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
        
        self.labels = best_labels
        return best_labels


class OptimizedHierarchicalClustering:
    """优化的层次聚类：Ward + 动态树截断"""
    def __init__(self, n_clusters, linkage_method='ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
    
    def fit_predict(self, X):
        """执行层次聚类"""
        Z = linkage(X, method=self.linkage_method)
        labels = fcluster(Z, self.n_clusters, criterion='maxclust') - 1
        return labels


class GMMClustering:
    """高斯混合模型：概率聚类"""
    def __init__(self, n_clusters, n_init=10, max_iter=200, random_state=42):
        self.n_clusters = n_clusters
        self.gmm = GaussianMixture(n_components=n_clusters, n_init=n_init, 
                                    max_iter=max_iter, random_state=random_state)
    
    def fit_predict(self, X):
        """GMM聚类"""
        self.gmm.fit(X)
        return self.gmm.predict(X)


class DBSCANClustering:
    """DBSCAN密度聚类：自适应eps参数"""
    def __init__(self, eps=None, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def _estimate_eps(self, X):
        """使用k-distance图自动估计eps"""
        from sklearn.neighbors import NearestNeighbors
        k = self.min_samples
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = np.sort(distances[:, -1], axis=0)
        # 取距离分布的90百分位数
        eps = np.percentile(distances, 90)
        return eps
    
    def fit_predict(self, X):
        """DBSCAN聚类"""
        if self.eps is None:
            eps = self._estimate_eps(X)
        else:
            eps = self.eps
        
        db = DBSCAN(eps=eps, min_samples=self.min_samples)
        labels = db.fit_predict(X)
        # 处理噪声点：将-1标签分配到最近的簇
        if (labels == -1).sum() > 0:
            from sklearn.neighbors import NearestNeighbors
            valid_mask = labels != -1
            if valid_mask.sum() > 0:
                nbrs = NearestNeighbors(n_neighbors=1).fit(X[valid_mask])
                distances, indices = nbrs.kneighbors(X[labels == -1])
                labels[labels == -1] = labels[valid_mask][indices.ravel()]
        return labels


class HDBSCANClustering:
    """HDBSCAN：层次密度聚类"""
    def __init__(self, min_cluster_size=10, min_samples=None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
    
    def fit_predict(self, X):
        """HDBSCAN聚类"""
        if not HAS_HDBSCAN:
            raise ImportError("HDBSCAN not installed. Install with: pip install hdbscan")
        
        clusterer = HDBSCAN(min_cluster_size=self.min_cluster_size, 
                           min_samples=self.min_samples)
        labels = clusterer.fit_predict(X)
        # 处理噪声点
        if (labels == -1).sum() > 0:
            from sklearn.neighbors import NearestNeighbors
            valid_mask = labels != -1
            if valid_mask.sum() > 0:
                nbrs = NearestNeighbors(n_neighbors=1).fit(X[valid_mask])
                distances, indices = nbrs.kneighbors(X[labels == -1])
                labels[labels == -1] = labels[valid_mask][indices.ravel()]
        return labels


class EnsembleClusteringVoting:
    """集成聚类：多个聚类算法投票"""
    def __init__(self, n_clusters, algorithms=None, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        if algorithms is None:
            self.algorithms = [
                ('kmeans', ImprovedKMeans(n_clusters, random_state=random_state)),
                ('hierarchical', OptimizedHierarchicalClustering(n_clusters)),
                ('gmm', GMMClustering(n_clusters, random_state=random_state)),
            ]
        else:
            self.algorithms = algorithms
    
    def fit_predict(self, X):
        """执行集成聚类，通过KMeans投票合并结果"""
        results = []
        for name, alg in self.algorithms:
            try:
                pred = alg.fit_predict(X)
                # 验证聚类数量合理性
                n_unique = len(np.unique(pred[pred >= 0]))
                if n_unique >= 2:
                    results.append(pred)
            except Exception as e:
                print(f"Algorithm {name} failed: {e}")
        
        if not results:
            print("All algorithms failed, using KMeans fallback")
            return ImprovedKMeans(self.n_clusters, random_state=self.random_state).fit_predict(X)
        
        # 将预测结果堆叠并用KMeans在聚类分配空间中进行最终聚类
        stacked = np.column_stack(results)
        ensemble_clusterer = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        final_labels = ensemble_clusterer.fit_predict(stacked)
        return final_labels


def create_feature_matrix(X, method='pca', n_components=50):
    """创建特征矩阵（禁用自编码器，使用PCA、NMF等）"""
    # 处理NaN值
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'pca':
        pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
        return pca.fit_transform(X_scaled)
    elif method == 'nmf':
        from sklearn.decomposition import NMF
        nmf = NMF(n_components=min(n_components, X_scaled.shape[1]-1), init='random', random_state=42)
        # NMF需要非负输入
        X_nonneg = X_scaled - X_scaled.min() + 1e-10
        return nmf.fit_transform(X_nonneg)
    else:
        return X_scaled


def run_all_methods(X, true_labels, n_clusters, feature_method='pca'):
    """运行所有创新聚类方法并返回结果"""
    # 特征提取
    print(f"特征提取（方法：{feature_method}）...")
    X_features = create_feature_matrix(X, method=feature_method, n_components=50)
    
    results = {}
    
    # 1. 改进的K-means
    print("运行改进K-means...")
    kmeans_improved = ImprovedKMeans(n_clusters)
    pred = kmeans_improved.fit_predict(X_features)
    results['Improved_KMeans'] = {'pred': pred, 'metrics': evaluate_clustering(true_labels, pred)}
    print(f"  NMI={results['Improved_KMeans']['metrics']['NMI']:.4f}")
    
    # 2. 核K-means
    print("运行核K-means...")
    kernel_kmeans = KernelKMeans(n_clusters)
    pred = kernel_kmeans.fit_predict(X_features)
    results['Kernel_KMeans'] = {'pred': pred, 'metrics': evaluate_clustering(true_labels, pred)}
    print(f"  NMI={results['Kernel_KMeans']['metrics']['NMI']:.4f}")
    
    # 3. 层次聚类
    print("运行优化层次聚类...")
    hier = OptimizedHierarchicalClustering(n_clusters)
    pred = hier.fit_predict(X_features)
    results['Hierarchical'] = {'pred': pred, 'metrics': evaluate_clustering(true_labels, pred)}
    print(f"  NMI={results['Hierarchical']['metrics']['NMI']:.4f}")
    
    # 4. GMM
    print("运行高斯混合模型...")
    gmm = GMMClustering(n_clusters)
    pred = gmm.fit_predict(X_features)
    results['GMM'] = {'pred': pred, 'metrics': evaluate_clustering(true_labels, pred)}
    print(f"  NMI={results['GMM']['metrics']['NMI']:.4f}")
    
    # 5. DBSCAN
    print("运行DBSCAN...")
    dbscan_method = DBSCANClustering(min_samples=int(np.sqrt(len(X_features))))
    pred = dbscan_method.fit_predict(X_features)
    n_unique = len(np.unique(pred[pred >= 0]))
    if n_unique >= 2:
        results['DBSCAN'] = {'pred': pred, 'metrics': evaluate_clustering(true_labels, pred)}
        print(f"  NMI={results['DBSCAN']['metrics']['NMI']:.4f}")
    
    # 6. HDBSCAN（如果可用）
    if HAS_HDBSCAN:
        print("运行HDBSCAN...")
        try:
            hdbscan_method = HDBSCANClustering(min_cluster_size=max(10, len(X_features)//100))
            pred = hdbscan_method.fit_predict(X_features)
            n_unique = len(np.unique(pred[pred >= 0]))
            if n_unique >= 2:
                results['HDBSCAN'] = {'pred': pred, 'metrics': evaluate_clustering(true_labels, pred)}
                print(f"  NMI={results['HDBSCAN']['metrics']['NMI']:.4f}")
        except Exception as e:
            print(f"  HDBSCAN failed: {e}")
    
    # 7. 集成聚类
    print("运行集成聚类...")
    ensemble = EnsembleClusteringVoting(n_clusters)
    pred = ensemble.fit_predict(X_features)
    results['Ensemble'] = {'pred': pred, 'metrics': evaluate_clustering(true_labels, pred)}
    print(f"  NMI={results['Ensemble']['metrics']['NMI']:.4f}")
    
    return results, X_features


if __name__ == '__main__':
    print("创新聚类方法模块已加载。")
