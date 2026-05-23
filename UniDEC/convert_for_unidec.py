import json
import os

def convert_labels(file_in, file_out):
    """把逗号分隔换成空格分隔，写入到 UniDEC 需要的标签文件"""
    with open(file_in, 'r') as fin, open(file_out, 'w') as fout:
        for line in fin:
            line = line.strip()
            if line:  # 避免空行
                # 将逗号或制表符统一替换为空格
                label_ids = line.replace(',', ' ').replace('\t', ' ')
                fout.write(label_ids + '\n')

def extract_text_from_json(json_file, text_file):
    """从 trn.json / tst.json 中提取文本，写入每行一个文本"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 数据可能是 [{"title": "...", ...}, ...] 或 {"data": [...]}，按需调整
    # 这里假设为一个 list，每个元素有 'title' 字段
    if isinstance(data, dict):
        # 如果是一个字典，尝试找到包含数据的列表
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                data = v
                break
    with open(text_file, 'w', encoding='utf-8') as f:
        for item in data:
            # 优先尝试 'title'，然后是 'text'，然后是 'sentence'
            text = item.get('title') or item.get('text') or item.get('sentence') or ''
            f.write(text.strip() + '\n')

if __name__ == '__main__':
    data_dir = '.'  # 脚本放在数据目录里，直接使用当前路径
    # 1. 转换标签文件
    convert_labels(os.path.join(data_dir, 'filter_labels_train'), os.path.join(data_dir, 'trn_X_Y.txt'))
    convert_labels(os.path.join(data_dir, 'filter_labels_test'), os.path.join(data_dir, 'tst_X_Y.txt'))

    # 2. 提取文本文件
    extract_text_from_json(os.path.join(data_dir, 'trn'), os.path.join(data_dir, 'train_texts.txt'))
    extract_text_from_json(os.path.join(data_dir, 'tst'), os.path.join(data_dir, 'test_texts.txt'))

    print("转换完成！生成的文件：trn_X_Y.txt, tst_X_Y.txt, train_texts.txt, test_texts.txt")
