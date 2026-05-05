import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.optim import AdamW

from dataset import MDataset, createDataCSV
from model import XMLMHE
from utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--data_path', type=str, required=False, default='/home/user/Data')
parser.add_argument('--bert', type=str, required=False, default='bert-base')
parser.add_argument('--bert_path', type=str, required=False, default='../NLP-Model/')

# LoRA 相关参数
parser.add_argument('--use_lora', action='store_true', help='use LoRA for Qwen model')
parser.add_argument('--lora_rank', type=int, default=256, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=256, help='LoRA alpha')
parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
# 混合精度
parser.add_argument('--bf16', action='store_true', help='use bfloat16 mixed precision')

parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--num_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')

args = parser.parse_args()


def train(model, df, label_map):
    
    tokenizer = model.get_tokenizer()

    print('dataset = ', args.dataset)
    train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len, group_y=group_y,
                        candidates_num=args.group_y_candidate_num,model_name=args.bert)
    test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, group_y=group_y,
                        candidates_num=args.group_y_candidate_num,model_name=args.bert)

    train_d.tokenizer = model.get_fast_tokenizer()
    test_d.tokenizer = model.get_fast_tokenizer()

    trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                shuffle=True)
    testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                            shuffle=False)
    if args.valid:
        print('valid ...')
        valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len, group_y=group_y,
                            candidates_num=args.group_y_candidate_num, model_name=args.bert)
        validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=0,
                                    shuffle=False)
   
    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']

    # For Qwen models, use different learning rates for backbone and classification head
    # Backbone (bert): args.lr, Classification head (l0, l1, embed): args.lr * 2
    if 'qwen' in args.bert.lower():
        backbone_params = []
        backbone_params_no_decay = []
        head_params = []
        head_params_no_decay = []

        for n, p in model.named_parameters():
            if 'bert' in n:
                # Backbone parameters (including LoRA)
                if any(nd in n for nd in no_decay):
                    backbone_params_no_decay.append(p)
                else:
                    backbone_params.append(p)
            else:
                # Classification head parameters (l0, l1, embed, etc.)
                if any(nd in n for nd in no_decay):
                    head_params_no_decay.append(p)
                else:
                    head_params.append(p)

        optimizer_grouped_parameters = [
            {'params': backbone_params, 'weight_decay': 0.01, 'lr': args.lr},
            {'params': backbone_params_no_decay, 'weight_decay': 0.0, 'lr': args.lr},
            {'params': head_params, 'weight_decay': 0.01, 'lr': args.lr * 2},
            {'params': head_params_no_decay, 'weight_decay': 0.0, 'lr': args.lr * 2},
        ]
        print(f'Using differential learning rates: backbone={args.lr}, head={args.lr * 2}')
    else:
        # For other models (BERT, RoBERTa, XLNet), use unified learning rate
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        print(f'Using unified learning rate: {args.lr}')

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)


    max_only_p5 = -1
    for epoch in range(0, args.epoch): ##
        train_loss = model.train_epoch(epoch, trainloader, optimizer, mode='train',
                                     eval_loader=validloader if args.valid else testloader,
                                     eval_step=args.eval_step, log=LOG)

        if args.valid:
            ev_result = model.eval_epoch(epoch, validloader)
        else:
            ev_result = model.eval_epoch(epoch, testloader)

        g_p1, g_p3, g_p5, p1, p3, p5 = ev_result

        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, train_loss:{train_loss}'
        if args.num_group>0:
            log_str += f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}'
        if args.valid:
            log_str += ' valid'
        LOG.log(log_str)

        if max_only_p5 < p5:
            max_only_p5 = p5
            model.save_model(f'models/model-{exp_name}.bin')

        if epoch >= args.epoch + 5 and max_only_p5 != p5:
            break

if __name__ == '__main__':
    init_seed(args.seed)
    exp_name = get_exp_name(args.dataset,args.bert,args.num_group)
    LOG = Logger('log_'+exp_name)
    
    print(f'load {args.dataset} dataset...')
    df, label_map = createDataCSV(args.dataset,path=args.data_path)
    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    if args.num_group>0:
        num_classes, per_ele_classes = check_gorup(len(label_map), args.num_group)
        group_y = get_groups_v2(len(label_map), num_classes, per_ele_classes)
    else:
        group_y = None
    model = XMLMHE(n_labels=len(label_map), group_y=group_y, bert=args.bert,
                        bert_path=args.bert_path,update_count=args.update_count,
                        use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                        candidates_topk=args.group_y_candidate_topk,
                        hidden_dim=args.hidden_dim)
    
    if args.eval_model and args.dataset in ['wiki500k', 'amazon670k','amazon3m']:
        print(f'load models/model-{exp_name}.bin ......')
        tokenizer = model.get_tokenizer()
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, group_y=group_y,
                           candidates_num=args.group_y_candidate_num, model_name=args.bert)

        test_d.tokenizer = model.get_fast_tokenizer()

        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                                shuffle=False)

        model.load_state_dict(torch.load(f'models/model-{exp_name}.bin',map_location='cpu'))
        model = model.cuda()

        pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
        np.save(f'results/{exp_name}-labels.npy', np.array(pred_labels))
        np.save(f'results/{exp_name}-scores.npy', np.array(pred_scores))
    else:
        train(model, df, label_map)
