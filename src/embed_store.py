from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# This is the folder where Chroma will store the database on your disk

def create_vector_store(chunks):

    # 1. Initialize the Embedding Model
    # We use 'nomic-embed-text' because it is optimized for retrieval
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    print(f"--- Processing {len(chunks)} chunks ---")

    # 2. Create or Update the Vector Store
    # Chroma is smart:
    # - If 'persist_directory' has data, it loads it.
    # - If not, it embeds the chunks and saves them there.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )



    # 3. Return the Retriever
    # This is the 'search engine' tool that puuu.py will use.
    # k=4 means "Give me the top 4 best matches"
    return vectorstore.as_retriever(search_kwargs={"k": 6})

"""

*************************************
EMBEDDINGS MODULE TESTED AND VERIFIED
*************************************


# --- TEST BLOCK ---
# Run this file directly to verify it works
if __name__ == "__main__":
    from load_pdf import load_pdf_file
    from chunker import split_documents
    import shutil  # Used to clean up old DB for testing

    test_pdf = "D:\RAG LANGCHAIN PROJECT\data\data_file111.pdf"

    # Optional: Clear old database for a fresh test
    # if os.path.exists(DB_PATH):
    #     shutil.rmtree(DB_PATH)

    if os.path.exists(test_pdf):
        print("1. Loading PDF...")
        docs = load_pdf_file(test_pdf)

        print("2. Splitting Text...")
        chunks = split_documents(docs)

        print("3. Creating Chroma Database...")
        retriever = create_vector_store(chunks)

        # Test the search
        query = "What is Version control ?"
        print(f"\n--- Testing Search for: '{query}' ---")
        results = retriever.invoke(query)

        if results:
            print(f" Success! Found {len(results)} relevant results.")
            print(f"Preview: {results[0].page_content[:100]}...")
        else:
            print(" No results found.")

"""
