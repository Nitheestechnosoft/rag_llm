from langchain_ollama import ChatOllama #Oolamachat to import llama 3
from langchain_core.prompts import ChatPromptTemplate #Chatpromttemplate is a function to create a template that adds question and context from the user
from langchain_core.output_parsers import StrOutputParser #
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # <--- Added RunnableLambda, it binds the other functions to each other /Runnable Passthrough splits the data to database and to the prompt
from langchain_community.retrievers import BM25Retriever #bestmatch25 for keyword search

# Custom modules
from src.load_pdf import load_pdf_file
from src.chunker import split_documents
from src.embed_store import create_vector_store


def run_rag_pipeline(pdf_path):
    print("---   Initializing Manual Hybrid RAG Pipeline ---")

    # 1. Load & Split
    docs = load_pdf_file(pdf_path)
    if not docs: return None
    chunks = split_documents(docs)

    # 2. Build Retrievers
    # A. Vector Search
    print("--- Building Vector Index ---")
    base_store = create_vector_store(chunks)
    vector_retriever = base_store.vectorstore.as_retriever(search_kwargs={"k":4})

    # B. Keyword Search
    print("---  Building Keyword Index ---")
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 5

    # 3. MANUAL HYBRID SEARCH LOGIC
    def hybrid_search(query):
        # Run both searches
        vector_docs = vector_retriever.invoke(query)
        keyword_docs = keyword_retriever.invoke(query)

        # Combine
        all_docs = vector_docs + keyword_docs
        unique_docs = []
        seen_content = set()

        for doc in all_docs:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)

        return unique_docs

    # 4. Initialize LLM
    llm = ChatOllama(model="llama3", temperature=0)

    # 5. Prompt
    template = """You are an expert analyst. Answer the question based ONLY on the context below.

    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 6. Format Function
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # 7. Build Chain (Fixed with RunnableLambda)
    # We wrap the functions so they can "talk" to each other using |
    rag_chain = (
            {
                "context": RunnableLambda(hybrid_search) | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    # Update this path to your actual file
    pdf_path = r"D:\RAG LANGCHAIN PROJECT\data\data_file.pdf"

    chain = run_rag_pipeline(pdf_path)

    if chain:
        print("\n" + "=" * 50)
        print(" HYBRID BOT READY! (Type 'exit' to quit)")
        print("=" * 50 + "\n")

        while True:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit"]: break
            try:
                response = chain.invoke(query)
                print(f"\nBot:\n{response}")
            except Exception as e:
                print(f"Error: {e}")
            print("-" * 50)
