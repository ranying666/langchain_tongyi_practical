from langchain_community.llms import Tongyi
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate

# Importing chat specific components
from langchain.memory import ConversationBufferMemory

# Template for chatbot's behavior
template = """
你是一个对用户没有帮助的聊天机器人。
你的目标是不帮助用户而只是开玩笑。
听取用户所说的话并开个玩笑

{chat_history}
Human: {human_input}
"""

# Creating a prompt template with input variables
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=template
)

# Initializing ConversationBufferMemory to store chat history
memory = ConversationBufferMemory(memory_key="chat_history")

# Creating an LLMChain with Tongyi language model
llm_chain = LLMChain(
    llm=Tongyi(),
    prompt=prompt,
    verbose=True,
    memory=memory
)


while True:  
    user_input = input("请输入你的消息（或输入 'exit' 退出）：")  
    if user_input.lower() == "exit":  
        break  
      
    # 使用聊天机器人回复用户的消息  
    response = llm_chain.run(human_input=user_input, chat_history="")  
    print(f"AI: {response}")
