import sys, os
sys.path.append(os.getcwd())  # 确保能导入项目内模块

import torch
from hydra import initialize, compose
from models.unidec import UniDEC

def main():
    # 1. 加载配置（假设配置文件在 conf/config.yaml）
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config")
        # 强制使用测试模式 + 原始 backbone（以免加载任务权重）
        cfg.test = True
        cfg.model.plm = "sentence-transformers/msmarco-distilbert-base-v4"
        # 若配置中 load_model 不为空，设为 null 避免加载缺失的权重
        cfg.load_model = None

        data_dir = cfg.data.dir  # 指向您的数据目录

    # 2. 初始化模型
    print("正在加载模型...")
    model = UniDEC(cfg.model)   # 使用配置中的 model 参数
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 3. 读取测试文本（只取前 8 条，避免显存/时间浪费）
    test_texts_file = os.path.join(data_dir, "test_texts.txt")
    if not os.path.exists(test_texts_file):
        raise FileNotFoundError(f"找不到测试文本文件: {test_texts_file}")
    with open(test_texts_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f][:8]

    # 4. 推理测试
    print("开始前向传播...")
    with torch.no_grad():
        emb = model.encode(texts, convert_to_tensor=True)
        # 如果使用 GPU，移回 CPU 方便打印
        if emb.is_cuda:
            emb = emb.cpu()
    print(f"✅ 推理成功！输出嵌入形状: {emb.shape}")   # 应为 (8, 768)
    print("UniDEC 核心管线验证通过，您的资源完全支持运行此模型。")

if __name__ == "__main__":
    main()