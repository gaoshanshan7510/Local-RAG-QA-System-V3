# 🤖 自动化与机电设备智能问答系统 (Local-RAG-Expert)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-orange.svg)
![BM25](https://img.shields.io/badge/BM25-Hybrid_Retrieval-purple.svg)
![BGE](https://img.shields.io/badge/BGE-Reranker-yellow.svg)

## 📖 项目简介
本项目是一个基于**纯本地大模型（Local LLM）**与 **RAG（检索增强生成）** 架构构建的垂直领域知识库问答系统。
系统致力于解决通用大模型在专业机电/自动化设备领域"幻觉"严重的问题，通过外挂本地知识库，为业务人员提供精准、可溯源的技术解答，并支持多轮自然对话。

> 🆕 **第三代迭代**：在第一代基础上，引入 BM25+Chroma **双路混合检索** 与 **BGE Reranker 精准重排**，大幅提升复杂专业问题的检索召回率与答案精准度。

---

## ✨ 核心特性

- **🔒 数据零泄露 (100% Local)**：基于 Ollama 本地部署 Qwen2.5 (7B) 大模型，无需调用外部 API，保障企业机密文档绝对安全。

- **🧠 意图识别与查询重写 (Query Rewrite)**：引入 LangChain 的 `history_aware_retriever` 机制。针对用户在多轮对话中的代词指代（如"那它的缺点是什么？"），大模型会结合历史上下文进行语义补全，大幅提升向量检索的准确率。

- **⚡ 极速响应与状态流转 (Performance)**：前端基于 Streamlit 构建。通过 `@st.cache_resource` 实现了大模型与向量数据库的单例模式（Singleton）驻留内存；结合 `session_state` 拦截并转换 UI 状态，实现前后端数据无缝流转与毫秒级响应。

- **🔀 双路混合检索 (Hybrid Retrieval)**：同时启用 **BM25 关键词检索**（精准字面匹配）与 **Chroma 向量语义检索**（深层语义理解），通过 `EnsembleRetriever` 以 RRF 算法融合两路各 Top-10 召回结果，解决单一检索策略的盲区问题。集成 `jieba` 分词，显著提升中文 BM25 检索精度。

- **🏆 BGE Reranker 精准重排 (Reranking)**：引入 `BAAI/bge-reranker-base` 交叉编码器，对混合召回的候选文档进行深度语义打分与重排，从粗筛的 20 个候选中精准提取 Top-3 最相关片段送入 LLM，大幅降低噪音干扰，并在 UI 界面实时展示每段文档的 BGE 相关性得分。

- **🎯 检索调优 (Top-K Tuning)**：通过对 Chroma 向量数据库进行切片策略测试与 Top-K 参数调优，在保证召回率的同时，有效控制上下文窗口，避免边缘噪音数据导致的 LLM 幻觉或崩溃。

---

## 🛠️ 技术栈

| 模块 | 技术选型 |
|------|------|
| 核心框架 | LangChain LCEL |
| 大语言模型 | Qwen2.5:7b（Ollama 本地驱动）|
| 文本嵌入模型 | nomic-embed-text（Ollama）|
| 向量数据库 | Chroma DB |
| 关键词检索 | BM25 + jieba 中文分词 |
| 重排模型 | bge-reranker-base |
| 前端交互 | Streamlit |

---

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.8+ 并启动 Ollama 服务：

```bash
ollama run qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 放入你的知识库
将需要检索的机电设备说明书、维修手册（支持 .txt 格式）放入 `my_knowledge_docs` 文件夹中。

### 4. 构建知识库索引（首次运行必须执行）

```bash
python rag_core.py
```
⚠️ 此步骤会同时生成 Chroma 向量数据库与 BM25 所需的 splits.pkl 文件，跳过此步将无法启动系统。

### 5. 运行系统

```bash
streamlit run app.py
```

在浏览器中打开 `http://localhost:8501` 即可开始与你的专属 AI 专家对话！

## 📂 项目结构 

```mermaid
graph TD
    Root[Local-RAG-Expert 项目] --> A[app.py : 前端UI与状态管理]
    Root --> B[rag_core.py : RAG核心逻辑]
    Root --> C[(my_knowledge_docs/ : 本地PDF知识库)]
    Root --> D[(chroma_db/ : 向量数据库+BM25索引)]
    Root --> E[requirements.txt : 依赖包]
    Root --> F[README.md : 说明文档]

    B --> B1[双路混合检索 BM25+Chroma]
    B --> B2[BGE Reranker 精准重排]
    B --> B3[历史感知查询重写]

    style Root fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#f66,stroke-width:2px,stroke-dasharray: 5 5
    style D fill:#bbf,stroke:#f66,stroke-width:2px,stroke-dasharray: 5 5
