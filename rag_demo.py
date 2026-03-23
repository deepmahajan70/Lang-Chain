# ==============================
# RAG DEMO (Latest LangChain)
# ==============================

# ---------- IMPORTS ----------
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from transformers import pipeline


# ---------- STEP 1: LOAD DATA ----------
def load_documents(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents


# ---------- STEP 2: SPLIT TEXT ----------
def split_documents(documents):
    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


# ---------- STEP 3: CREATE VECTOR DB ----------
def create_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)
    return db


# ---------- STEP 4: CREATE RETRIEVER ----------
def create_retriever(db):
    return db.as_retriever(search_kwargs={"k": 3})


# ---------- STEP 5: LOAD LLM ----------
def load_llm():
    pipe = pipeline(
        "text-generation",
        model="gpt2",   # lightweight for demo
        max_new_tokens=100
    )
    return HuggingFacePipeline(pipeline=pipe)


# ---------- STEP 6: BUILD RAG CHAIN (LCEL) ----------
def build_rag_chain(retriever, llm):

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain


# ---------- MAIN FUNCTION ----------
def main():

    # Step 0: Create sample data
    sample_text = """
    Industry 4.0 refers to the integration of IoT, cloud computing, and cyber-physical systems in manufacturing.

    Intrusion Detection Systems (IDS) are used to detect malicious activities in a network.

    Machine learning techniques like Isolation Forest and Autoencoders are commonly used in IDS.

    Precision inversion occurs when precision decreases due to majority attack traffic.
    """

    with open("data.txt", "w") as f:
        f.write(sample_text)

    # Step 1
    documents = load_documents("data.txt")

    # Step 2
    docs = split_documents(documents)

    # Step 3
    db = create_vector_db(docs)

    # Step 4
    retriever = create_retriever(db)

    # Step 5
    llm = load_llm()

    # Step 6
    rag_chain = build_rag_chain(retriever, llm)

    # Step 7: Query loop
    print("\n===== RAG SYSTEM READY =====\n")

    while True:
        query = input("Enter your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        response = rag_chain.invoke(query)

        print("\nAnswer:\n", response)
        print("\n" + "="*50 + "\n")


# ---------- RUN ----------
if __name__ == "__main__":
    main()