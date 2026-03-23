from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

cm = ChatGoogleGenerativeAI(model='gemini-3-flash-preview')
res = cm.invoke("Suggest 3 Indian boy's names")

print(res)