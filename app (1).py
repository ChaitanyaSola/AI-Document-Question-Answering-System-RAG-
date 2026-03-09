import gradio as gr

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline


# Load model once (important for HuggingFace Spaces)
generator = pipeline(
    "text-generation",
    model="google/flan-t5-small"
)

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def rag_system(pdf_file, text_input, question):

    documents = []

    # Load PDF
    if pdf_file is not None:
        loader = PyPDFLoader(pdf_file.name)
        pdf_docs = loader.load()
        documents.extend(pdf_docs)

    # Load text input
    if text_input.strip() != "":
        documents.append(Document(page_content=text_input))

    if len(documents) == 0:
        return "Please upload a PDF or enter some text."

    # Split text into chunks
    splitter = CharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    # Create vector database
    db = FAISS.from_documents(docs, embeddings)

    # Retrieve relevant chunks
    results = db.similarity_search(question, k=2)

    context = " ".join([doc.page_content for doc in results])

    prompt = f"""

Question:
{question}

Answer:
{context}

"""

    result = generator(prompt, max_length=200)

    return result[0]["generated_text"]


interface = gr.Interface(
    fn=rag_system,
    inputs=[
        gr.File(label="Upload PDF (optional)"),
        gr.Textbox(label="Or paste document text"),
        gr.Textbox(label="Ask your question")
    ],
    outputs="text",
    title="AI Document Question Answering System (RAG)"
)

interface.launch()