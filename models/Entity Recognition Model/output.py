import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer
from AlBertFusionAttCRF import AlBertFusionAttCRFTokenClassModel  # 确保自定义模型实现正确


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


if __name__ == "__main__":
    # 路径需与实际存储位置一致
    recognizer = EntityRecognizer("/root/autodl-tmp/wa1/result/model.pt")

    # 测试案例（与图片一致）
    test_case = "熊希龄、文俊铎在醴陵姜湾（今醴陵市）创办公立湖南瓷业学堂，从日本聘请教员。"
    print("实体识别结果:", recognizer.predict(test_case))