import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

#todo 1:定义函数,返回检索器
def get_retriever():
	#1.定义Embeddings
	embedding = OpenAIEmbeddings()
	#2.向量数据库持久化路径
	persist_directory ="./data_base/vector_db/chroma"
	#3.加载数据库
	vectordb = Chroma(
		persist_directory=persist_directory,
		embedding_function=embedding
	)
	return vectordb.as_retriever()

#todo 2: 定义combine函数,处理检索器返回的文本
def combine_docs(docs):
	return "\n\n".join(doc.page_content for doc in docs["context"])

#todo 3:定义get_qa_history_chain函数,返回一个检索问答链
def get_qa_history_chain():
	retriever = get_retriever()
	llm = ChatOpenAI(
		temperature=0.0,
		model=os.getenv("LLM_MODEL_ID"),
		api_key=os.getenv("LLM_API_KEY"),
		base_url=os.getenv("LLM_BASE_URL")
	)
	condense_question_system_template = (
		"请根据聊天记录总结客户最近的问题."
		"如果没有多余的聊天记录则返回用户的问题."
	)
	condense_question_prompt = ChatPromptTemplate(
		[
			("system",condense_question_system_template),
			("placeholder","{chat_history}"),
			("human","{input}")
		]
	)
#RunnableBranch(
	# (条件1,动作1),(条件2,动作2),....,兜底动作(前面都不执行才执行,前面执行了兜底动作不执行)
	# )
	retrieve_docs = RunnableBranch(
		(lambda x:not x.get("chat_history",False),(lambda x:x["input"])|retriever,),
		condense_question_prompt|llm|StrOutputParser()|retriever,
	)
	system_prompt = (
		"你是一个问答任务的助手."
		"请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
	)
	qa_prompt = ChatPromptTemplate.from_messages(
		[
			("system", system_prompt),
			("placeholder", "{chat_history}"),
			("human", "{input}"),
		]
	)
	qa_chain = (
		RunnablePassthrough().assign(context = combine_docs)
		|qa_prompt
		|llm
		|StrOutputParser()
	)
	qa_history_chain = RunnablePassthrough().assign(
		context = retrieve_docs
	).assign(answer = qa_chain)
	return qa_history_chain
	# todo: 这是一个重要的运行步骤说明，
	#  第一步:把所有的invoke()内容传给qa_history_chain(大字典),保持不动的同时,复制一个大字典大字典传给.assign(context = retrieve_docs)
	#  第二步(分支一):如果没有chat_history,只需要把input传给检索器,
	#  第二步(分支二):在retrieve_docs里如果有chat_history,就把整个复制来的大字典传给condense_question_prompt然后一步步传给检索器|llm|StrOutputParser()|retriever,
	#  第三步:将检索器返回的三个Document装进context,并把它加进到原本的大字典里
	#       此时大字典变为{"input": "...", "chat_history": [...], "context": [Document1, Document2, Document3]},
	#  第四步:将大字典再次复制传给qa_chain,qa_chain里将复制的大字典传给combine_docs函数,函数在docs["context"]里将三个Document连接成一个超级大文本,
	#       此时复制的大字典变为{"input": "...", "chat_history": [...], "context": "这里是拼接好的长篇大论文本..."},
	#  第五步:按照qa_prompt|llm||StrOutputParser()把复制的大词典扔进去,qa_prompt进行填空,把context填到system_prompt里赋值给system,stroutputparser返回纯对话文本内容,并将对话内容作为answer加入原本的大字典里,
	#       此时原本的大字典变为{"input": "...", "chat_history": [...], "context": [Document1, Document2, Document3],"answer": "大模型生成的最终文本回答"},
	#  第六步:返回大字典qa_history_chain

#todo 4:定义gen_response函数,，它接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def gen_response(chain,input,chat_history):
	response = chain.stream({   #流式输出用stream替代invoke
		"input":input,
		"chat_history":chat_history
	})
	for res in response:
		if "answer" in res: #当捕捉到"answer"再返回
			yield res["answer"] #生成器替代return,一个字一个字的返回

#todo 5:定义main函数,该函数制定显示效果与逻辑
def main():
	st.markdown('### 🦜🔗 动手学大模型应用开发')
	## st.session_state可以存储用户与应用交互期间的状态与数据
	# 1. 检查有没有聊过天，没有就创建一个空列表当“记事本”
	if "messages" not in st.session_state:
		st.session_state.messages = []
	#存储检索问答链
	# 2. 检查有没有把 RAG 链初始化过，没有的话在内存里加载一次（避免每次发话都重新读向量数据库，省时间！）
	if "qa_history_chain" not in st.session_state:
		st.session_state.qa_history_chain = get_qa_history_chain()
	#建立一个固定高度的聊天滚动容器,高度为500 px
	messages = st.container(height=500)
	# 核心：每次网页刷新，都要把以前聊过的话从头到尾重新画一遍！显示整个对话历史
	for message in st.session_state.messages:
		# messages指在容器下显示，chat_message显示用户及ai头像
		# message[0] 是 "human" 或 "ai"
		#with 的作用：它在告诉 Streamlit：“从现在开始到缩进结束，接下来所有的 UI 组件（比如这句 st.write(prompt)），
		#           全部给我打包塞进‘用户（human）’的那个聊天气泡框里去！”
		with messages.chat_message(message[0]):
			st.write(message[1])#message[1]是打印对话具体文字内容内容

		if prompt :=st.chat_input("Say something"):
			# 将用户输入添加到对话历史中
			st.session_state.messages.append(("human",prompt))
			# 显示当前用户输入
			with messages.chat_message("human"):
				st.write("prompt")
			answer = gen_response(
				chain = st.session_state.qa_history_chain,
				input = prompt,
				chat_history = st.session_state.messages
			)
			with messages.chat_message("ai"):
				output = st.write_stream(answer)
				st.session_state.messages.append(("ai",output))
