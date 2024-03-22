from langchain_community.llms import Tongyi
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """
Question: {question}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"])

print(prompt)

llm = Tongyi()

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "OKR关键结果的常见误区有哪些？"

res = llm_chain.run(question)

print(res)