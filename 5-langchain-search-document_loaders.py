'''
文档加载器
https://python.langchain.com/docs/integrations/document_loaders
'''
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import Html2TextTransformer

# Load HTML
# %pip install --upgrade --quiet  playwright beautifulsoup4
# ! playwright install
loader = AsyncChromiumLoader(["https://dataea.cn/okr-keyresult-checklist/"])
html = loader.load()


html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(html)

from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Tongyi

# Define prompt
prompt_template = """请给以下内容写一个摘要:
"{text}"
摘要:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = Tongyi()
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

print(stuff_chain.run(docs_transformed))