from transformers import BertModel
import torch.nn as nn
import numpy as np
import torch
import bert_config
import json
from torch.utils.data import Dataset, DataLoader, RandomSampler
# 这里要显示的引入BertFeature，不然会报错
from dataset import ReDataset
from preprocess import BertFeature
from preprocess import get_out, Processor

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()

        # 线性变换
        Q = self.query(x)  # (batch_size, seq_len, hidden_size)
        K = self.key(x)
        V = self.value(x)

        # 拆成多头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, V)

        # 合并多头
        context = context.transpose(1,2).contiguous().view(batch_size, seq_len, hidden_size)
        output = self.out(context)
        return output

class BertForRelationExtractionV2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = self.bert.config
        self.dropout = nn.Dropout(args.dropout_prob)

        # 实体位置编码
        self.position_emb = nn.Embedding(args.max_seq_len, self.bert_config.hidden_size)

        # 多头注意力
        self.multihead_attention = MultiHeadSelfAttention(self.bert_config.hidden_size, num_heads=8)

        # 跨实体交互模块
        self.cross_entity_attention = nn.MultiheadAttention(embed_dim=self.bert_config.hidden_size, num_heads=4)

        # 特征归一化
        self.layer_norm = nn.LayerNorm(self.bert_config.hidden_size * 6)

        # 分类器
        self.classifier = nn.Linear(self.bert_config.hidden_size * 6, 14)

    def forward(self, token_ids, attention_masks, token_type_ids, entity_positions):
        bert_outputs = self.bert(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids,
        )
        seq_out = bert_outputs.last_hidden_state

        # 多头自注意力
        seq_out = self.multihead_attention(seq_out)

        # 位置编码
        pos_emb = self.position_emb(entity_positions).sum(dim=1)

        batch_size = seq_out.size(0)
        entity_features = []

        for i in range(batch_size):
            positions = entity_positions[i]

            # 获取实体局部上下文
            e1_start, e1_end, e2_start, e2_end = positions
            e1_context = seq_out[i, max(0, e1_start - 2):min(seq_out.size(1), e1_end + 2)]
            e2_context = seq_out[i, max(0, e2_start - 2):min(seq_out.size(1), e2_end + 2)]

            # 池化局部特征
            e1_feature = e1_context.mean(dim=0)
            e2_feature = e2_context.mean(dim=0)

            # 跨实体交互
            cross_feature, _ = self.cross_entity_attention(
                e1_feature.unsqueeze(0).unsqueeze(0),  # [1,1,hidden]
                e2_feature.unsqueeze(0).unsqueeze(0),
                e2_feature.unsqueeze(0).unsqueeze(0)
            )
            cross_feature = cross_feature.squeeze(0).squeeze(0)

            # 特征组合
            combined = torch.cat([
                e1_feature,
                e2_feature,
                e1_feature * e2_feature,
                e1_feature - e2_feature,
                pos_emb[i],
                cross_feature,
            ])
            entity_features.append(combined)

        features = torch.stack(entity_features)
        features = self.layer_norm(features)
        features = self.dropout(features)
        return self.classifier(features)