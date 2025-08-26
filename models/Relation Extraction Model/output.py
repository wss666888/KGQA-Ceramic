from pprint import pprint
import os
import logging
import json
import shutil
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader, RandomSampler
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
        # Load pre-trained model and optimizer from the checkpoint
        self.checkpoint_path = '/root/autodl-tmp/wa2/checkpoints/best.pt'
        self.model, self.optimizer, _, _ = self.load_ckp(self.model, self.optimizer, self.checkpoint_path)

    def load_ckp(self, model, optimizer, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        """保存检查点"""
        torch.save(state, checkpoint_path)

    def test(self):
        """测试模型"""
        self.model.eval()
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                ids = test_data['ids'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids, ids)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(), axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())
        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, id2label, args, ids):
        """单条样本预测"""
        self.model.eval()
        with torch.no_grad():
            # 对文本进行编码
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
                outputs = [id2label[i] for i in outputs]  # 转换为标签文本
                return outputs
            else:
                return '未识别出关系'

    def get_metrics(self, outputs, targets):
        """计算评估指标"""
        accuracy = accuracy_score(targets, outputs)
        micro_f1 = f1_score(targets, outputs, average='micro')
        macro_f1 = f1_score(targets, outputs, average='macro')
        return accuracy, micro_f1, macro_f1


def get_entity_token_positions(tokenizer, text):
    # 替换标记符号
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

    # 数据预处理
    processor = preprocess.Processor()

    # 加载标签映射
    label2id = {}
    id2label = {}
    with open('./data/rel_dict.json', 'r', encoding="utf-8") as fp:
        labels = json.loads(fp.read())
    for k, v in labels.items():
        label2id[k] = v
        id2label[v] = k

    # 准备测试数据
    test_out = preprocess.get_out(processor, './data/test.txt', args, id2label, 'test')
    test_features, test_callback_info = test_out
    test_dataset = dataset.ReDataset(test_features)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.eval_batch_size,
                             num_workers=2)

    # 初始化训练器（不需要训练和验证数据加载器）
    trainer = Trainer(args, test_loader=test_loader)

    # 测试模型
    logger.info('========进行测试========')
    total_loss, test_outputs, test_targets = trainer.test()
    accuracy, micro_f1, macro_f1 = trainer.get_metrics(test_outputs, test_targets)
    logger.info(f"【test】 loss：{total_loss:.6f} accuracy：{accuracy:.4f} micro_f1：{micro_f1:.4f} macro_f1：{macro_f1:.4f}")

    # 进行单条预测
    logger.info('========单条预测示例========')
    tokenizer = BertTokenizerFast.from_pretrained(args.bert_dir)
    sample_text = " $五铢堂窑址$位于#浙江省金华市金东区塘雅镇五渠塘村东郊五渠塘水库底#。"
    entity_ids = get_entity_token_positions(tokenizer, sample_text)
    prediction = trainer.predict(tokenizer, sample_text, id2label, args, entity_ids)
    logger.info(f"预测文本：{sample_text}")
    logger.info(f"预测关系：{prediction}")
    logger.info(f"真实标签：{id2label[4]}")
