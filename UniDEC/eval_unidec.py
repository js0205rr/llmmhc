import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from types import SimpleNamespace
from models.unidec import UniDEC

# 设置离线模式（防止联网）
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def load_labels(file_path):
    """读取标签文件，每行是空格分隔的标签ID，返回 list of list of int"""
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append([int(x) for x in line.split()])
    return labels

def load_texts(file_path):
    """读取文本文件，每行一个文本"""
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f]
    return texts

def precision_at_k(pred_ids, true_ids, k=5):
    """计算单个样本的 Precision@k"""
    if len(pred_ids) < k:
        pred_ids = pred_ids + [pred_ids[-1]] * (k - len(pred_ids))  # 补齐
    pred_set = set(pred_ids[:k])
    true_set = set(true_ids)
    return len(pred_set & true_set) / k

def ndcg_at_k(pred_ids, true_ids, k=5):
    """计算单个样本的 NDCG@k"""
    dcg = 0.0
    idcg = 0.0
    true_set = set(true_ids)
    for i, pid in enumerate(pred_ids[:k], 1):
        relevance = 1.0 if pid in true_set else 0.0
        dcg += relevance / np.log2(i + 1)
    for i in range(min(k, len(true_ids))):
        idcg += 1.0 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

def main():
    # === 1. 配置（沿用之前成功的模式） ===
    cfg = OmegaConf.load("config.yaml")
    hardware_dict = OmegaConf.to_container(cfg.hardware, resolve=True)
    model_dict = OmegaConf.to_container(cfg.model, resolve=True)
    required_additions = {
        "dataset": "LF-AmazonTitles-131K",
        "num_labels": 131073,
        "num_proc": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "hidden_dims": 384,
        "contrastive_dims": 384,
        "shortlist_size": 100,
        "temp": 0.07,
        "lr": 1e-4,
        "CLF_loss": "bce",
        "DE_loss": "supcon",
        "add_dual_loss": False,
        "add_dual_clf_loss": False,
        "add_bce_loss": False,
        "mixed_precision": hardware_dict.get("mixed_precision", "bf16"),
        "test": True,
        "load_model": "",
        # plm 指向本地模型文件夹
        "plm": "msmarco-distilbert-base-v4",
    }
    model_dict.update(required_additions)
    cfg_model = SimpleNamespace(**model_dict)

    # === 2. 数据路径 ===
    data_dir = "/notebook/dataset/Amazon131k"
    test_texts = load_texts(os.path.join(data_dir, "test_texts.txt"))
    test_labels = load_labels(os.path.join(data_dir, "tst_X_Y.txt"))
    train_texts = load_texts(os.path.join(data_dir, "train_texts.txt"))
    train_labels = load_labels(os.path.join(data_dir, "trn_X_Y.txt"))

    # === 3. 初始化模型（随机权重） ===
    print("加载模型...")
    model = UniDEC(cfg_model)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        print("已启用 GPU")

    # === 4. 编码训练集和测试集（分批） ===
    BATCH_SIZE = 32   # 可根据显存调整
    def encode_texts(model, texts, desc="编码中"):
        all_emb = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
            batch = texts[i:i+BATCH_SIZE]
            with torch.no_grad():
                emb = model.bert.encode(batch, convert_to_tensor=True)
                if emb.is_cuda:
                    emb = emb.cpu()
                all_emb.append(emb.numpy())
        return np.concatenate(all_emb, axis=0)

    print("编码训练集...")
    train_emb = encode_texts(model, train_texts, "训练集编码")
    print("编码测试集...")
    test_emb = encode_texts(model, test_texts, "测试集编码")

    # === 5. 标签矩阵（构建标签索引） ===
    # 统计总标签数（根据数据集为131073）
    num_classes = 131073
    # 构建训练集的二值标签矩阵（稀疏形式直接转为每个样本的标签列表已经够用）
    # 对于检索，我们简单使用最近邻方式：将测试样本嵌入与所有训练样本嵌入做点积，
    # 得到相似度，然后综合训练样本的标签来推荐标签（这是一个简化的多标签检索方法）
    # 这里采用“标签平均嵌入”的常见做法：对每个标签，计算拥有该标签的所有训练样本嵌入的中心
    print("构建标签中心...")
    label_centers = np.zeros((num_classes, train_emb.shape[1]), dtype=np.float32)
    label_counts = np.zeros(num_classes, dtype=np.int32)
    for i, lbl_list in enumerate(train_labels):
        for lbl in lbl_list:
            if lbl < num_classes:
                label_centers[lbl] += train_emb[i]
                label_counts[lbl] += 1
    # 避免除零
    for lbl in range(num_classes):
        if label_counts[lbl] > 0:
            label_centers[lbl] /= label_counts[lbl]

    # === 6. 检索与评估（Precision@k, nDCG@k） ===
    k = 5
    prec_list = []
    ndcg_list = []
    print("评估中...")
    # 将标签中心归一化便于计算点积（可选）
    test_emb_norm = test_emb / np.linalg.norm(test_emb, axis=1, keepdims=True)
    centers_norm = label_centers / np.linalg.norm(label_centers, axis=1, keepdims=True)
    # 点积相似度
    similarity = np.dot(test_emb_norm, centers_norm.T)   # [n_test, num_classes]
    # 取 top-k 标签索引
    topk_indices = np.argsort(-similarity, axis=1)[:, :k]   # [n_test, k]

    for i, true_lbls in enumerate(test_labels):
        pred_ids = topk_indices[i].tolist()
        prec = precision_at_k(pred_ids, true_lbls, k=k)
        ndcg = ndcg_at_k(pred_ids, true_lbls, k=k)
        prec_list.append(prec)
        ndcg_list.append(ndcg)

    avg_prec = np.mean(prec_list)
    avg_ndcg = np.mean(ndcg_list)
    print(f"\n===== 评估结果（随机权重，无训练） =====")
    print(f"Precision@{k}: {avg_prec:.4f}")
    print(f"NDCG@{k}: {avg_ndcg:.4f}")
    print("注意：指标很低是因为分类头未训练，仅验证流程可行性。")

if __name__ == "__main__":
    main()
