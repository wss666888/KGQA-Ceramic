import pandas as pd
import numpy as np
from collections import Counter
from transformers import BertTokenizer

data = ['古', '代', '御', '窑', '、', '官', '窑', '以', '及', '民', '窑', '中', '，', '大', '都', '以', '手', '工', '做',
        '陶','瓷','器','为','主','。']
# data = data + ['[PAD]'] * 128
# data = data[:128]
print(data)
Tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/wa1/bert-base-chinese')
print(Tokenizer)
c = Tokenizer.encode_plus(text=data,
                          padding='max_length',
                          max_length=128,
                          return_tensors='pt')

print(c)
print(c['attention_mask'].shape)
