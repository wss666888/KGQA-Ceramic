import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
from AlBertFusionAttCRF import AlBertFusionAttCRFTokenClassModel  # 确保自定义模型实现正确
import os

class EntityRecognizer:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 关键修复1：修正预训练模型名称（移除多余下划线）
        self.tokenizer = BertTokenizer.from_pretrained("albert_-chinese-base")

        # 关键修复2：标准化标签体系（与训练数据完全一致）
        self.label_list = [
            'O', 'B-陶瓷器类', 'I-陶瓷器类', 'B-陶瓷原料', 'I-陶瓷原料',
            'B-陶瓷文博', 'I-陶瓷文博', 'B-陶瓷文献', 'I-陶瓷文献', 'B-窑业遗存',
            'I-窑业遗存', 'B-陶瓷装饰', 'I-陶瓷装饰', 'B-陶瓷人物', 'I-陶瓷人物',
            'B-陶瓷教育机构', 'I-陶瓷教育机构', 'B-地点', 'I-地点',
            'B-历史时期', 'I-历史时期'
        ]

        # 关键修复3：统一模型名称定义
        self.model = AlBertFusionAttCRFTokenClassModel('albert_-chinese-base').to(self.device)
        self._load_model_weights(model_path)
        self.model.eval()

    def _load_model_weights(self, model_path):
        """完整参数加载逻辑"""
        try:
            pretrained_dict = torch.load(model_path, map_location=self.device)
        except FileNotFoundError:
            raise FileNotFoundError(f"模型路径错误: {model_path}")

        # 关键修复4：增强参数名映射
        adapted_dict = {}
        for k, v in pretrained_dict.items():
            new_key = k.replace("albert.encoder", "bertModel.encoder") \
                .replace("albert.embeddings", "bertModel.embeddings") \
                .replace("albert.", "bertModel.")
            adapted_dict[new_key] = v

        # 关键修复5：严格加载模式诊断
        load_info = self.model.load_state_dict(adapted_dict, strict=True)
        print(f"参数加载诊断：\n缺失参数: {load_info.missing_keys}\n意外参数: {load_info.unexpected_keys}")

    def convert_to_tensors(self, text):
        """输入处理标准化"""
        # 关键修复6：统一参数名（attention_mask → input_mask）
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "input_mask": encoded["attention_mask"].to(self.device),  # 参数名对齐模型定义
            "lstm_array": torch.zeros(1, 128, 200).to(self.device)
        }

    def predict(self, text):
        """预测流程优化"""
        # 获取模型输入
        inputs = self.convert_to_tensors(text)

        # 获取预测结果
        with torch.no_grad():
            _, predictions = self.model(**inputs)

        # 从 input_ids 中还原 tokens（必须）
        input_ids = inputs["input_ids"][0].cpu().numpy().tolist()
        mask = inputs["input_mask"].cpu().numpy().squeeze()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 去除[CLS]和[SEP]，获取有效 tokens 和预测标签
        valid_indices = np.where(mask == 1)[0][1:-1]
        tokens = [tokens[i] for i in valid_indices]

        pred_ids = predictions[0].cpu().numpy().flatten().tolist()
        valid_pred_ids = pred_ids[1:len(valid_indices) + 1]
        valid_labels = [self.label_list[i] if i < len(self.label_list) else 'O'
                        for i in valid_pred_ids]

        # BIO标签修复
        corrected = []
        prev_tag = 'O'
        for label in valid_labels:
            if label.startswith('I-'):
                entity_type = label.split('-', 1)[1]
                if prev_tag in [f'B-{entity_type}', f'I-{entity_type}']:
                    corrected.append(label)
                else:
                    corrected.append(f'B-{entity_type}')
            else:
                corrected.append(label)
            prev_tag = corrected[-1]
        return self._decode_entities(corrected, tokens)

    def _decode_entities(self, labels, tokens):
        """实体解析优化"""
        entities = []
        current = None

        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                if current:
                    entities.append(current)
                current = {'type': label[2:], 'tokens': [token]}
            elif label.startswith('I-'):
                if current and current['type'] == label[2:]:
                    current['tokens'].append(token)
                else:  # 处理非连续I标签
                    if current:
                        entities.append(current)
                    current = {'type': label[2:], 'tokens': [token]}
            else:
                if current:
                    entities.append(current)
                    current = None

        if current:
            entities.append(current)

        return [(e['type'], ''.join(e['tokens'])) for e in entities]
    def predict_with_marked_text(self, text):
        """返回实体识别结果及标注实体后的文本"""
        entities = self.predict(text)

        # 优先选择前两个实体作为标注目标
        marked_text = text
        if len(entities) >= 2:
            # 为了避免位置混乱，先按实体在文本中出现的位置进行排序（从后往前插入）
            ent1_text = entities[0][1]
            ent2_text = entities[1][1]

            # 确保两个实体都出现在原文中
            if ent1_text in marked_text and ent2_text in marked_text:
                # 先替换后出现的，防止位置混乱
                if marked_text.index(ent1_text) > marked_text.index(ent2_text):
                    ent1_text, ent2_text = ent2_text, ent1_text

                marked_text = marked_text.replace(ent2_text, f"#{ent2_text}#", 1)
                marked_text = marked_text.replace(ent1_text, f"${ent1_text}$", 1)

        return entities, marked_text



if __name__ == "__main__":

    # ==== 模型路径 ====
    recognizer = EntityRecognizer("/root/autodl-tmp/wa1/result/model.pt")

    # ==== 文件路径 ====
    input_file = "input_texts.txt"           # 输入文本
    output_marked_file = "marked_texts.txt"  # 输出：用于关系抽取的标注文本
    output_entity_file = "entities_info.txt" # 输出：实体识别日志

    # ==== 批量处理 ====
    if not os.path.exists(input_file):
        print(f"❌ 未找到输入文件：{input_file}")
    else:
        with open(input_file, "r", encoding="utf-8") as fin, \
             open(output_marked_file, "w", encoding="utf-8") as fout_marked, \
             open(output_entity_file, "w", encoding="utf-8") as fout_entity:

            for line_id, line in enumerate(fin, 1):
                text = line.strip()
                if not text:
                    continue

                # 实体识别 + 标注处理
                entities, marked_text = recognizer.predict_with_marked_text(text)

                # 写入标注结果（供关系抽取使用）
                fout_marked.write(marked_text + "\n")

                # 写入实体详细信息
                fout_entity.write(f"[第{line_id}行]\n原文：{text}\n")
                fout_entity.write("识别实体：\n")
                for ent_type, ent_text in entities:
                    fout_entity.write(f" - {ent_text} ({ent_type})\n")
                fout_entity.write(f"标注文本：{marked_text}\n")
                fout_entity.write("-" * 50 + "\n")

                print(f"✅ 已处理第{line_id}行：{marked_text}")
