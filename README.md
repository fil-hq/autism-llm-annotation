# autism-llm-annotation
# 自闭症论文研究：标注代码与分类手册仓库说明（README）

本仓库用于存放自闭症在线社区/在线问诊语料的主题与身份标注相关代码、分类手册与提示词等资料。

## 目录结构

- `Code/`：用于存放标注代码  
  - `siliconflow_mark.py`：使用硅基智能（SiliconFlow）API 进行标注  
  - `vllm_mark.py`：使用本地模型 + vLLM 服务进行标注  
- `Codebook/`：分类手册（Codebook）  
  - `Codebook_en/`：英文版分类手册  
  - `Codebook_ch/`：中文版分类手册  
- `Info/`：辅助信息与运行所需文件  
  - `prompt.xlsx`：项目使用的提示词（Prompts）  
  - `requirement.txt`：运行环境所需依赖库列表（requirements）

## 模型与推理方式说明

- 标注模型中，**除 671B 模型外**，其余模型均为**本地下载的模型文件**并在本地推理运行（配合 vLLM）。
- **671B 模型**由于体量原因，标注时使用 **硅基智能（SiliconFlow）API** 进行推理。

## 分类手册说明（Codebook）

`Codebook/` 文件夹包含身份（poster_role）与主题（topic / primary_topic / secondary_topic）的详细分类规范，涵盖：
- 各标签的操作化定义（definitions）
- 常见关键词线索（keywords）
- 示例记录（examples）
- 混淆类区分规则（disambiguation rules）

---

# Autism Paper Project: Annotation Code & Codebook Repository (README)

This repository stores code and documentation for annotating **topics** and **poster identities** in autism-related online community and online medical consultation corpora.

## Repository Structure

- `Code/`: Annotation scripts  
  - `siliconflow_mark.py`: Annotation via **SiliconFlow (硅基智能) API**  
  - `vllm_mark.py`: Annotation via **local models served with vLLM**  
- `Codebook/`: Classification codebook  
  - `Codebook_en/`: English codebook  
  - `Codebook_ch/`: Chinese codebook  
- `Info/`: Supporting files  
  - `prompt.xlsx`: Prompts used in this project  
  - `requirement.txt`: Python dependencies required to run the code

## Models & Inference Setup

- For annotation, **all models except the 671B model** are run using **locally downloaded model files** (served and inferred locally via vLLM).
- The **671B model** is annotated using the **SiliconFlow API** due to its scale.

## Codebook Description

The `Codebook/` folder provides detailed specifications for classifying **poster identities** (`poster_role`) and **topics** (`topic / primary_topic / secondary_topic`), including:
- Operational definitions of labels
- Keyword cues
- Example records
- Confusion-class disambiguation rules

