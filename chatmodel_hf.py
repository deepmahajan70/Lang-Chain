import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# 1. Load your API key from .env file
load_dotenv()

# 2. Setup the Endpoint with a forced provider
# 'hf-inference' uses Hugging Face's own infrastructure which 
# supports 'text-generation' for Llama-3.
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

myllm = HuggingFaceEndpoint(
    repo_id="bigscience/bloom-560m",
    task="text2text-generation"
)

# 3. Use the Llama-3 Prompt Template
# Text-generation models need the structure to know where to start answering.
prompt = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Suggest 3 Indian boy's names.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

try:
    print(f"Connecting to {repo_id} via hf-inference...")
    res = myllm.invoke(prompt)
    print("\n--- Model Response ---")
    print(res)
except Exception as e:
    print(f"\nFailed to connect: {e}")