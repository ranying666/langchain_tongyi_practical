from langchain.chains import APIChain
from langchain_community.llms import Tongyi

llm = Tongyi(temperature=0)

api_docs = """

BASE URL: https://apiv6.sofreight.com/

API Documentation:

API 接口 api/Tools/airPortCountry?country_code={name}&keyword={keyword} 用于查找有关国家/地区的机场信息。 下面列出了所有 URL 参数：
     - name：国家/地区代码 - 例如：FR,CN,US
     - keyword: 关键词，当有关键词的时候不使用国家代码
    
"""

chain_new = APIChain.from_llm_and_api_docs(llm, api_docs, verbose=True, limit_to_domains=["https://apiv6.sofreight.com/"])


# 通过国家名称来获取信息
#chain_new.run('你能告诉我有关法国的机场信息吗?')
#chain_new.run('你能告诉我有比利时的机场信息吗?')
# 通过关键字来获取信息
chain_new.run('你能告诉我有关北京的机场信息吗?')
