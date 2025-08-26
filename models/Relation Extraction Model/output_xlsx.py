from pprint import pprint
import os
import logging
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
import bert_config
import preprocess
import dataset
import models
from utils.utils import set_seed, set_logger


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, args, train_loader=None, dev_loader=None, test_loader=None):
        self.args = args
        gpu_ids = args.gpu_ids.split(',')
        self.device = torch.device("cpu" if gpu_ids[0] == '-1' else "cuda:" + gpu_ids[0])
        self.model = models.BertForRelationExtraction(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)
        self.checkpoint_path = '/root/autodl-tmp/wa2/checkpoints/best.pt'
        self.model, self.optimizer, _, _ = self.load_ckp(self.model, self.optimizer, self.checkpoint_path)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def predict(self, tokenizer, text, id2label, args, ids):
        self.model.eval()
        with torch.no_grad():
            inputs = tokenizer.encode_plus(
                text=text,
                add_special_tokens=True,
                max_length=args.max_seq_len,
                truncation='longest_first',
                padding="max_length",
                return_token_type_ids=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            token_ids = inputs['input_ids'].to(self.device).long()
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            ids = torch.from_numpy(np.array([[x + 1 for x in ids]])).to(self.device)
            outputs = self.model(token_ids, attention_masks, token_type_ids, ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten().tolist()
            if len(outputs) != 0:
                outputs = [id2label[i] for i in outputs]
                return outputs
            else:
                return '未识别出关系'

    def save_predictions_to_excel(self, predictions, filename):
        df = pd.DataFrame(predictions, columns=["实体1", "实体2", "关系", "文本"])
        df.to_excel(filename, index=False)


def get_entity_token_positions(tokenizer, text):
    text = text.replace('#', '[E1]').replace('$', '[E2]')
    encode = tokenizer(text, return_offsets_mapping=True, max_length=128, padding='max_length', truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(encode['input_ids'])
    e1_pos = [i for i, tok in enumerate(tokens) if tok == '[E1]']
    e2_pos = [i for i, tok in enumerate(tokens) if tok == '[E2]']
    if len(e1_pos) >= 2 and len(e2_pos) >= 2:
        return [e1_pos[0], e1_pos[1], e2_pos[0], e2_pos[1]]
    else:
        return [1, 2, 3, 4]


if __name__ == '__main__':
    # 初始化配置
    args = bert_config.Args().get_parser()
    set_seed(args.seed)
    set_logger(os.path.join(args.log_dir, '/root/autodl-tmp/wa2/logs/main.log'))

    # 加载标签映射
    label2id = {}
    id2label = {}
    with open('./data/rel_dict.json', 'r', encoding="utf-8") as fp:
        labels = json.loads(fp.read())
    for k, v in labels.items():
        label2id[k] = v
        id2label[v] = k

    # 初始化 Trainer 和 tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)
    trainer = Trainer(args)

    # 开始处理 marked_texts.txt
    logger.info('========开始读取 marked_texts.txt 并进行关系预测========')
    predictions = []
    with open('./marked_texts.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 获取实体位置编码信息
        entity_ids = get_entity_token_positions(tokenizer, line)

        # 转换模型可识别的特殊标记
        text_for_model = line.replace('$', '[E2]').replace('#', '[E1]')

        # 模型预测
        prediction = trainer.predict(tokenizer, text_for_model, id2label, args, entity_ids)

        # 实体抽取
        try:
            e1 = line.split('$')[1]
        except:
            e1 = '未识别'
        try:
            e2 = line.split('#')[1]
        except:
            e2 = '未识别'

        relation = prediction[0] if prediction != '未识别出关系' else '未识别出关系'
        predictions.append([e1, e2, relation, line])

    # 保存结果
    trainer.save_predictions_to_excel(predictions, '三元组.xlsx')
    logger.info("✅ 三元组保存成功，文件名为 '三元组.xlsx'")
