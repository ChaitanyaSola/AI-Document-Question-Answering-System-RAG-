import os
import gradio as gr

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# API KEY (from HF secrets)

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# LOAD MODEL

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# GLOBAL VECTOR DB
db = None



# PROCESS PDF
def process_pdf(pdf_file):
    global db

    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embedding)

    return "✅ PDF processed successfully!"



# RAG FUNCTION

def ask_question(question):
    global db

    if db is None:
        return "⚠️ Please upload and process a PDF first."

    if not question:
        return "Please ask a question."

    results = db.similarity_search(question, k=2)

    context = "\n\n".join([doc.page_content for doc in results])

#     prompt = f"""
# You are a resume parser. Extract information ONLY from the context.
# Do NOT add, guess, or invent any information.

# Context:
# {context}

# Question: {question}

# List ONLY the exact items mentioned in the context.
# Answer:
# """
    prompt = f"""
You are an intelligent document assistant.

Your job is to analyze the given document and answer the user's question accurately.

Rules:
1. Use ONLY the provided context.
2. Do NOT add, assume, or hallucinate any information.
3. If the answer is not found, respond: "Not found in the document."
4. Be clear, concise, and structured.
5. If the question asks for:
   - List → return bullet points
   - Summary → give short summary
   - Skills/keywords → return comma-separated values
   - Explanation → give simple explanation

Context:
{context}

User Question:
{question}

Answer:
"""

    response = model.invoke(prompt)

    return response.content


# -----------------------------
# GRADIO UI

with gr.Blocks() as app:
    gr.Markdown("# 📄 RAG Chatbot")

    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    upload_btn = gr.Button("Process PDF")

    status = gr.Textbox(label="Status")

    question = gr.Textbox(label="Ask a Question")
    ask_btn = gr.Button("Get Answer")

    output = gr.Textbox(label="Answer")

    upload_btn.click(
        process_pdf,
        inputs=pdf_input,
        outputs=status
    )

    ask_btn.click(
        ask_question,
        inputs=question,
        outputs=output
    )

# -----------------------------
# RUN APP

app.launch()
