'''
文档加载器
https://python.langchain.com/docs/integrations/document_loaders
'''
from langchain_community.document_loaders import ConfluenceLoader
from langchain_community.document_transformers import Html2TextTransformer

loader = ConfluenceLoader(
    url="http://c.x/", username="090173",  api_key="12345"
)
documents = loader.load(page_id="92411492", include_attachments=False, limit=50)

print(documents)

html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(documents)

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