import os
import pickle # 用于保存文档块供 BM25 使用 
import jieba  # 用于中文 BM25 分词 

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama

# --- 核心修改点 1：新导入了支持记忆功能的包 ---
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 核心修改点 2： 重排序 Reranker 所需的包 ---
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- 核心修改点 3： BM25 与 混合检索 (Ensemble) 所需的包 ---
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from typing import Sequence, Any, Optional
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.callbacks import Callbacks

# 1. 配置参数
DOC_PATH = "./my_knowledge_docs"  # PDF存放路径！
DB_PATH = "./chroma_db"           # 向量数据库保存路径
SPLITS_PATH = "./chroma_db/splits.pkl" # 切分文档保存路径 

llm = Ollama(model="qwen2.5:7b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

class ScoreInjectingReranker(BaseDocumentCompressor):
    model: Any
    top_n: int = 3

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        # 1. 将用户问题和粗筛出来的文档组装成对
        texts = [[query, doc.page_content] for doc in documents]
        
        # 2. 让 BGE 模型打分
        scores = self.model.score(texts)
        
        # 3. 把文档和分数绑定在一起
        docs_with_scores = list(zip(documents, scores))
        
        # 4. 按分数从高到低排序
        docs_with_scores.sort(key=lambda x: x[1], reverse=True)

        # 5. 提取前 N 个，并把分数强行塞进 metadata
        final_docs = []
        for doc, score in docs_with_scores[:self.top_n]:
            if score < 0.1:
              continue 
                
            new_metadata = doc.metadata.copy()
            new_metadata["relevance_score"] = float(score) 
            final_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))
            
        return final_docs

def jieba_preprocess(text: str) -> list[str]:
    """--- 新增：为 BM25 提供中文分词支持 ---"""
    return list(jieba.cut_for_search(text))

def build_vector_db():
    """读取PDF并构建向量数据库 & 存储BM25所需的文档块"""
    print("正在加载PDF文档...")
    loader = PyPDFDirectoryLoader(DOC_PATH)
    docs = loader.load()
    print("正在将文档切分为小块...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # --- 新增：保存 splits 供 BM25 初始化使用 ---
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
    with open(SPLITS_PATH, "wb") as f:
        pickle.dump(splits, f)
    print("文档块已保存本地 (BM25专用)！")

    print("正在生成向量并存入 Chroma 数据库 ...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_PATH)
    print("数据库构建完成！")
    return vectorstore

def get_qa_chain():
    """构建带有记忆功能 + 双路混合检索 + Rerank重排 的终极问答链"""
    
    # ================= 检索策略：双路召回 =================
    # 路 1：Chroma 向量检索 (语义召回)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 路 2：BM25 关键词检索 (字面精准召回)
    if not os.path.exists(SPLITS_PATH):
        raise FileNotFoundError("找不到 splits.pkl，请先运行 build_vector_db() 构建数据库！")
    
    with open(SPLITS_PATH, "rb") as f:
        splits = pickle.load(f)
        
    print("正在初始化 BM25 检索器...")
    bm25_retriever = BM25Retriever.from_documents(splits, preprocess_func=jieba_preprocess)
    bm25_retriever.k = 10 # 同样粗筛 10 个
    
    # 混合路：使用 EnsembleRetriever 将两者结合 (底层使用 RRF 算法合并排序)
    # weights 参数：决定两者的权重比例
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6] # 向量偏语义权重稍微给高点，可自行调节
    )

    # 加载 BGE 重排模型 
    print("正在加载 Reranker 重排模型（首次运行会自动下载，请耐心等待）...")
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    
    # 从双路召回的混合结果里，用 BGE 模型精准挑出最准的 3 个 
    compressor = ScoreInjectingReranker(model=model, top_n=3)

    # 这里的 base_retriever 换成双路混合的 ensemble_retriever 
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensemble_retriever 
    )
    
    contextualize_q_system_prompt = (
        "给定聊天历史记录和最新的用户问题，"
        "该问题可能会引用历史记录中的上下文，"
        "请制定一个不需要历史记录也能听懂的独立问题。"
        "注意：不要回答问题，只需在需要时重新表述，否则按原样返回。"
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    system_prompt = (
        "你是一个专业的航空航天与自动化设备工程师。请使用以下检索到的检索内容来回答用户的问题。"
        "如果你不知道答案，请直接说不知道，不要编造。回答要尽量专业、有条理。\n\n"
        "检索内容：\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

if __name__ == "__main__":
    if not os.path.exists(DB_PATH) or not os.path.exists(SPLITS_PATH):
        build_vector_db()
        
    chain = get_qa_chain()
    chat_history = [] 
    
    print("=========== 开始测试多轮对话 & 混合检索重排 ===========")
    
    # ---------------- 第一轮对话 ----------------
    question1 = "什么是闭环控制系统？" 
    print(f"\n👤 用户: {question1}")
    
    response1 = chain.invoke({
        "input": question1,
        "chat_history": chat_history 
    })
    answer1 = response1["answer"]
    
    print("\n🔍 [后台探秘] BM25+Chroma双路召回 -> Reranker打分的最强资料：")
    for i, doc in enumerate(response1.get("context", [])):
        score = doc.metadata.get('relevance_score', '未知') 
        print(f"  🏆 第{i+1}名 (BGE得分: {score}) -> {doc.page_content[:60]}...")
        
    print(f"\n🤖 AI回答: {answer1}")
    
    chat_history.extend([
        HumanMessage(content=question1),
        AIMessage(content=answer1)
    ])
    
    # ---------------- 第二轮对话 ----------------
    question2 = "那它的优点和缺点是什么？" 
    print(f"\n\n👤 用户 (故意追问): {question2}")
    
    response2 = chain.invoke({
        "input": question2,
        "chat_history": chat_history 
    })
    
    print("\n🔍 [后台探秘] BM25+Chroma双路召回 -> Reranker打分的最强资料：")
    # 打印 response2.get("context", [])
    for i, doc in enumerate(response2.get("context", [])):
        print(f"👉 来源 {i+1} 的全部隐藏属性: {doc.metadata}")
        score = doc.metadata.get('relevance_score', '未知') 
        print(f"  🏆 第{i+1}名 (BGE得分: {score}) -> {doc.page_content[:60]}...")
        
    print(f"\n🤖 AI回答: {response2['answer']}")
