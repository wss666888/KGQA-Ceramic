# -*- coding: utf-8 -*-
import json
import re
import pandas as pd
from pprint import pprint

# 读取 Excel 数据
df = pd.read_excel('关系抽取训练数据.xlsx')

# 创建关系字典
relations = list(df['关系'].unique())
relations.remove('记载于')
relation_dict = {'记载于': 0}
relation_dict.update(dict(zip(relations, range(1, len(relations) + 1))))

# 保存关系字典到 JSON 文件
with open('rel_dict.json', 'w', encoding='utf-8') as h:
    h.write(json.dumps(relation_dict, ensure_ascii=False, indent=2))

# 为数据添加关系 ID
df['rel'] = df['关系'].apply(lambda x: relation_dict[x])

# 初始化结果列表
res = []
i = 1

# 遍历每条数据
for per1, per2, text, label in zip(df['实体1'].tolist(), df['实体2'].tolist(), df['文本'].tolist(), df['rel'].tolist()):
    # 确保实体为字符串
    per1 = str(per1)
    per2 = str(per2)

    if per1 in per2:
        # 替换实体2为占位符
        text_tmp = text.replace(per2, '#' * (len(per2) + 2))
        # 替换实体1
        text_tmp = text_tmp.replace(per1, '#' + per1 + '#')
        print(text_tmp)
        # 恢复实体2并标注
        text_tmp = text_tmp.replace('#' * (len(per2) + 2), '$' + per2 + '$')
        res1 = re.search('#' + per1 + '#', text_tmp)
        res2 = re.search(r'\$' + per2 + r'\$', text_tmp)

        # 检查匹配
        if res1 is not None and res2 is not None:
            text = text_tmp + '\t' + str(res1.span()[0]) + '\t' + str(res1.span()[1] - 1) + '\t' + str(
                res2.span()[0]) + '\t' + str(res2.span()[1] - 1)
            print(text)
        else:
            print("匹配失败: ", per1, per2)
            continue

    elif per2 in per1:
        # 替换实体1为占位符
        text_tmp = text.replace(per1, '#' * (len(per1) + 2))
        # 替换实体2
        text_tmp = text_tmp.replace(per2, '$' + per2 + '$')
        print(text_tmp)
        # 恢复实体1并标注
        text_tmp = text_tmp.replace('#' * (len(per1) + 2), '#' + per1 + '#')
        res1 = re.search('#' + per1 + '#', text_tmp)
        res2 = re.search(r'\$' + per2 + r'\$', text_tmp)

        # 检查匹配
        if res1 is not None and res2 is not None:
            text = text_tmp + '\t' + str(res1.span()[0]) + '\t' + str(res1.span()[1] - 1) + '\t' + str(
                res2.span()[0]) + '\t' + str(res2.span()[1] - 1)
            print(text)
        else:
            print("匹配失败: ", per1, per2)
            continue

    else:
        # 直接替换实体1和实体2
        text = text.replace(per1, '#' + per1 + '#').replace(per2, '$' + per2 + '$')
        res1 = re.search('#' + per1 + '#', text)
        res2 = re.search(r'\$' + per2 + r'\$', text)

        # 检查匹配
        if res1 is not None and res2 is not None:
            text = text + '\t' + str(res1.span()[0]) + '\t' + str(res1.span()[1] - 1) + '\t' + str(
                res2.span()[0]) + '\t' + str(res2.span()[1] - 1)
        else:
            print("匹配失败: ", per1, per2)
            continue

    # 保存结果
    res.append([text, label])
    i += 1

# 转换为 DataFrame
df = pd.DataFrame(res, columns=['text', 'rel'])
df['length'] = df['text'].apply(lambda x: len(x))

# 过滤文本长度大于 128 的数据
df = df[df['length'] <= 128]

# 输出统计信息
print('总数: %s' % len(df))
pprint(df['rel'].value_counts())
pprint(df['length'].value_counts())

# 划分训练集和测试集
train_df = df.sample(frac=0.8, random_state=1024)
test_df = df.drop(train_df.index)

# 保存训练集和测试集到文件
with open('train.txt', 'w', encoding='utf-8') as f:
    for text, rel in zip(train_df['text'].tolist(), train_df['rel'].tolist()):
        f.write(str(rel) + '\t' + text + '\n')

with open('test.txt', 'w', encoding='utf-8') as g:
    for text, rel in zip(test_df['text'].tolist(), test_df['rel'].tolist()):
        g.write(str(rel) + '\t' + text + '\n')
