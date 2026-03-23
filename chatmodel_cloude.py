from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

cm = ChatAnthropic(model='claude-opus-4-6')
res = cm.invoke("Suggest 3 Indian boy's names")

print(res.content)