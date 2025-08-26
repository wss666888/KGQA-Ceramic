import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from torchcrf import CRF


class DomainEmbeddingLayer(nn.Module):
    """领域感知嵌入层，融合通用词汇和领域术语表示"""

    def __init__(self, config, domain_vocab_size, domain_embed_dim=128):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.domain_embeddings = nn.Embedding(domain_vocab_size, domain_embed_dim)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 将领域嵌入投影到BERT隐藏空间
        self.domain_projection = nn.Linear(domain_embed_dim, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, domain_ids, token_type_ids=None, position_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if position_ids is None:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # 获取各项嵌入
        words_embeds = self.word_embeddings(input_ids)
        domain_embeds = self.domain_embeddings(domain_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        position_embeds = self.position_embeddings(position_ids)

        # 投影领域嵌入到BERT空间并融合
        domain_projected = self.domain_projection(domain_embeds)
        embeddings = words_embeds + domain_projected + token_type_embeds + position_embeds

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class DomainBERTForTokenClassification(BertPreTrainedModel):
    """基于DomainBERT的序列标注模型"""

    def __init__(self, config, domain_vocab_size, num_labels):
        super().__init__(config)
        self.config = config
        self.num_labels = num_labels
        self.domain_vocab_size = domain_vocab_size

        # 领域感知嵌入
        self.embeddings = DomainEmbeddingLayer(config, domain_vocab_size)

        # BERT编码器
        self.encoder = BertModel(config).encoder

        # 分类头
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # CRF层
        self.crf = CRF(num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, domain_ids, attention_mask=None, labels=None, token_type_ids=None):
        # 生成嵌入
        embedding_output = self.embeddings(
            input_ids,
            domain_ids,
            token_type_ids=token_type_ids
        )

        # 编码
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_ids.size())
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask)
        sequence_output = encoder_outputs[0]  # [batch_size, seq_len, hidden_size]

        # 分类
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        outputs = (logits,)

        if labels is not None:
            # 使用CRF损失
            loss = -self.crf(logits, labels, mask=attention_mask.byte())
            outputs = (loss,) + outputs

        return outputs

    def neg_log_likelihood(self, input_ids, input_mask, domain_ids, label_ids):
        """负对数似然计算（兼容您的训练代码）"""
        return self.forward(
            input_ids=input_ids,
            domain_ids=domain_ids,
            attention_mask=input_mask,
            labels=label_ids
        )[0]