# Knowledge Graph Construction and Question Answering Applications in the Ceramic Domain
# 项目概述

本项目提出一套面向陶瓷领域的知识图谱构建与智能问答技术框架（该项目框架亦可应用于其他领域），实现以下功能：

## 核心功能

1. **结构化知识建模**  
   - 通过改进的 NER（命名实体识别）与 RE（关系抽取）模型  
   - 自动化抽取陶瓷领域实体及其关系  

2. **多跳语义推理**  
   - 支持复杂语义查询  
   - 示例：*"醴陵窑主要产品记录在哪些文献中？"*  

3. **动态知识更新**  
   - 增量式图谱更新机制  
   - 确保知识库信息时效性  

4. **多轮对话交互**  
   - 上下文感知的对话管理  
   - 支持连续、连贯的问答场景  

### 系统流程示意图

![系统示意图](https://github.com/user-attachments/assets/b0ead2ff-5a8b-411b-bb21-65722aec37f6)

## 核心技术

### 1. 陶瓷领域知识抽取

#### 命名实体识别（NER）
- **模型架构**：BERT-FUSIONATT-CRF  
- **技术特点**：融合 BERT 语义表示、动态注意力机制与 CRF 序列优化，实现高精度实体识别  

#### 关系抽取（RE）
- **创新模型**：CrossRE-BERT  
- **技术特点**：引入位置编码 + 实体交互注意力机制，提升实体关系抽取效果  

### 2. 知识图谱构建
- 基于上述 NER 与 RE 模型自动抽取实体与关系  
- 优化图谱存储结构，提高查询和推理效率
#### 知识图谱构建效果图
  <img width="421" height="320" alt="image" src="https://github.com/user-attachments/assets/61501dfc-a679-4141-81e6-57c562dd9d66" />

### 3. 智能问答系统

#### 核心机制

- **增量更新**：动态融合新知识，保障知识库实时性  
- **多轮对话**：上下文关联推理，实现连续、连贯问答
#### 问答系统示例展示

https://github.com/user-attachments/assets/05040c07-4a0a-4da4-905d-e61b8116f900





