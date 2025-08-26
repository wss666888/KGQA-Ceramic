from transformers import BertModel, BertTokenizer
import shutil
import os
# 下载并加载预训练的BERT模型和词汇表
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("voidful/albert_chinese_base")
model = AutoModelForMaskedLM.from_pretrained("voidful/albert_chinese_base")
'''
model_name = 'clue/albert_chinese_base'
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
'''

model.save_pretrained('../albert_-chinese-base')
tokenizer.save_pretrained('../albert_-chinese-base')