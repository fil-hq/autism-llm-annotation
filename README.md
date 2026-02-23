# LLM-Assisted Annotation Framework for Cross Platform Analysis of Autism Online Communities: Implications for Parent Education and Digital Support
# Annotation Code & Codebook Repository (README)

This repository stores code and documentation for annotating **topics** and **identities** in autism online community and online medical consultation corpora.

## Repository Structure

- `Code/`: Annotation scripts  
  - `siliconflow_mark_1.py`, `vllm_mark_1.py`: **Primary topic annotation** for **Baidu Tieba**  
  - `siliconflow_mark_2.py`, `vllm_mark_2.py`: **Scondary topic annotation** for **Baidu Tieba**
  - `siliconflow_mark_consultation.py`, `vllm_mark_consultation.py`: Annotation for **physician–patient
consultation platforms** (**Chunyu Doctor** and **Haodf**)  
- `Codebook/`: Classification codebook  
  - `Codebook_en/`: English codebook  
  - `Codebook_ch/`: Chinese codebook  
- `Info/`: Supporting files  
  - `prompt.xlsx`: Prompts used in this project  
  - `requirement.txt`: Python dependencies required to run the code

## Models & Inference Setup

- For annotation, **all models except the 671B model** are run using **locally downloaded model files** (served and inferred locally via vLLM).（https://modelscope.cn/）
- The **671B model** is annotated using the **SiliconFlow API** due to its scale.（https://cloud.siliconflow.cn/）

## Codebook Description

The `Codebook/` folder provides detailed specifications for classifying **identities**  and **topics** , including:
- Definition
- Keywords
- Example records

---

 # 标注代码与分类手册仓库说明（README）
本仓库用于存放自闭症在线社区/在线问诊语料的主题与身份标注相关代码、分类手册与提示词等资料。

## 目录结构

- `Code/`：用于存放标注代码  
  - `siliconflow_mark_1.py`、`vllm_mark_1.py`：用于 **百度贴吧（Tieba）一级主题** 标注  
  - `siliconflow_mark_2.py`、`vllm_mark_2.py`：用于 **百度贴吧（Tieba）二级主题** 标注
  - `siliconflow_mark_consultation.py`、`vllm_mark_consultation.py`：用于 **在线医疗问诊平台（春雨医生、好大夫）** 语料标注  
- `Codebook/`：分类手册（Codebook）  
  - `Codebook_en/`：英文版分类手册  
  - `Codebook_ch/`：中文版分类手册  
- `Info/`：辅助信息  
  - `prompt.xlsx`：项目使用的提示词（Prompts）  
  - `requirement.txt`：运行环境所需依赖库列表（requirements）

## 模型与推理方式说明

- 标注模型中，**除 671B 模型外**，其余模型均为**魔搭下载的模型文件**并在本地推理运行（配合 vLLM）（https://modelscope.cn/）
- **671B 模型**由于体量原因，标注时使用 **硅基智能（SiliconFlow）API** 进行推理（https://cloud.siliconflow.cn/）

## 分类手册说明（Codebook）

`Codebook/` 文件夹包含身份（identity）与主题（topic）的详细分类规范，涵盖：
- 各标签的操作化定义（Definition）
- 常见关键词线索（Keywords）
- 示例记录（Example records）

