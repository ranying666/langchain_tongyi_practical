# from getpass import getpass

# 实际上是要输入密码的方式输入Key
# DASHSCOPE_API_KEY = getpass()

# import os
# os.environ["DASHSCOPE_API_KEY"] = "sk-"

from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """
Question: {question}
Answer: 可以逐步思考并详细回答这个问题
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"])

print(prompt)

llm = Tongyi()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "怎样学习大模型并将其应用到自己的业务中？"

res = llm_chain.run(question)

print(res)