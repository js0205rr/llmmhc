import json

def extract_labels_from_jsonlines(input_jsonl, output_txt):
    with open(input_jsonl, 'r', encoding='utf-8') as fin, \
         open(output_txt, 'w') as fout:
        for line in fin:
            obj = json.loads(line.strip())
            # 使用 target_ind 字段，若没有则尝试 target_id 等
            labels = obj.get('target_ind') or obj.get('labels') or obj.get('targets') or []
            # 写入空格分隔的标签 ID
            fout.write(' '.join(str(int(l)) for l in labels) + '\n')

if __name__ == '__main__':
    extract_labels_from_jsonlines('trn.json', 'trn_X_Y.txt')
    extract_labels_from_jsonlines('tst.json', 'tst_X_Y.txt')
    print("标签文件生成完毕！请用 head 检查前两行。")