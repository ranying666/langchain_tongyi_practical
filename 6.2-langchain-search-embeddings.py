from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
'''
从网页中获取文本信息，并向量化存储到本地，使用Embedding进行嵌入，精确检索信息
'''
# Load HTML
loader = AsyncChromiumLoader(["https://dataea.cn/okr-keyresult-checklist/"])
html = loader.load()

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(html)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs_transformed)

# embeddings = OpenAIEmbeddings()
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1", dashscope_api_key="dashscope_api_key") # 替换dashscope_api_key

vector_stores = FAISS.from_documents(texts, embeddings)

# 默认情况下，向量存储检索器使用相似性搜索。
retriever = vector_stores.as_retriever(
    # 相似性分数阈值检索
#    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

prompt: str = """请使用下面提供的背景信息来回答最后的问题。 如果背景信息中没有答案，请直接说不知道，不要试图凭空编造答案。
回答时尽量完整。"
{context}
问题: {question}
有用的回答:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=prompt)

llm = Tongyi()

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
result = qa_chain({"query": "OKR关键结果的常见误区有哪些？"})

print(result.get("result"))