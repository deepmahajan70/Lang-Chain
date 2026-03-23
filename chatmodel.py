from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

cm = ChatOpenAI(model='gpt-5.4-nano',temperature=1,max_completion_tokens=10)
res = cm.invoke("Suggest 3 Indian boy's names")

print(res)