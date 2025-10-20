import os
import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ---------------------- CACHED BACKEND FUNCTIONS ----------------------

@st.cache_resource(show_spinner="📄 Processing your BPCL document...")
def create_vector_db(uploaded_file):
    """Creates a Chroma vector database from the uploaded BPCL PDF."""
    # Save uploaded file to a temporary path (Chroma / loaders expect a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        # Load & Split PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)
        documents = [Document(page_content=doc.page_content) for doc in docs]

        # Create Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Create Vector DB
        vector_db = Chroma.from_documents(documents, embedding=embeddings)
        return vector_db

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@st.cache_resource
def get_llm(api_key: str):
    """Initializes and returns the ChatOpenAI LLM.

    Note: this function assumes the `ChatOpenAI` wrapper accepts the shown keyword
    arguments (model, temperature, base_url, max_tokens, api_key). Adjust as
    needed for your specific langchain integration.
    """
    return ChatOpenAI(
        model="openai/gpt-3.5-turbo",
        temperature=0.2,
        base_url="https://openrouter.ai/api/v1",
        max_tokens=500,
        api_key=api_key
    )


def get_response(llm, retriever, chat_history, question: str) -> str:
    """Generates an answer using a simple RAG pipeline.

    This function builds a prompt from the retrieved context and chat history,
    then invokes the LLM wrapper to get a textual answer.
    """
    template = """You are a BPCL FAQ assistant.
Use the following pieces of retrieved context from the BPCL FAQ document to answer the question.
If the answer isn't clear, politely say you don't know. Keep answers concise and relevant.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""
    prompt_template = ChatPromptTemplate.from_template(template)

    chat_history_text = "No previous conversation."
    if chat_history:
        formatted = [f"{role.capitalize()}: {msg}" for role, msg in chat_history[-6:]]
        chat_history_text = "\n".join(formatted)

    # Retrieve relevant documents (defensive: support different retriever APIs)
    if retriever is None:
        return "No vector store available to retrieve context from."

    if hasattr(retriever, 'get_relevant_documents'):
        relevant_docs = retriever.get_relevant_documents(question)
    elif hasattr(retriever, 'retrieve'):
        relevant_docs = retriever.retrieve(question)
    elif hasattr(retriever, 'invoke'):
        relevant_docs = retriever.invoke(question)
    else:
        # As a last resort, attempt to call as a function
        try:
            relevant_docs = retriever(question)
        except Exception:
            relevant_docs = []

    context = "\n\n".join(getattr(doc, 'page_content', str(doc)) for doc in relevant_docs)

    formatted_prompt = prompt_template.format(
        context=context,
        chat_history=chat_history_text,
        question=question
    )

    # Call LLM (defensive for different interface shapes)
    if hasattr(llm, 'invoke'):
        result = llm.invoke(formatted_prompt)
        answer = getattr(result, 'content', str(result))
    elif hasattr(llm, 'generate'):
        result = llm.generate(formatted_prompt)
        # extract text if possible
        answer = getattr(result, 'text', str(result))
    else:
        # fallback
        answer = str(llm)

    return answer


# ---------------------- MAIN STREAMLIT APP ----------------------

st.set_page_config(page_title="BPCL NLP FAQ Chatbot", page_icon="⛽", layout="wide")

st.title("⛽ BPCL NLP FAQ Chatbot")
st.markdown("Upload your BPCL FAQ PDF and chat with it using AI!")

# --- SESSION STATE ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📤 Upload BPCL FAQ PDF", type=["pdf"])

    api_key = st.text_input(
        "🔑 Enter your OpenRouter API Key",
        type="password",
        help="Get a free API key from https://openrouter.ai/"
    )

    if api_key:
        st.success("✅ API Key Added!")
    else:
        st.warning("⚠️ Please enter your OpenRouter API key to start chatting.")

    if st.button("🗑️ Clear & Reset"):
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        try:
            create_vector_db.clear()
        except Exception:
            pass
        st.experimental_rerun()

# --- MAIN CHAT AREA ---
if uploaded_file:
    if st.session_state.vector_db is None:
        st.session_state.vector_db = create_vector_db(uploaded_file)
        st.success("✅ BPCL document processed! Start chatting below.")

    # Display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    if user_question := st.chat_input("Ask a question about BPCL policies or FAQs..."):
        if not api_key:
            st.warning("Please enter your API key in the sidebar.")
            st.stop()

        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                llm = get_llm(api_key)
                retriever = None
                if hasattr(st.session_state.vector_db, 'as_retriever'):
                    retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                else:
                    # try direct use of vector_db as retriever
                    retriever = st.session_state.vector_db

                answer = get_response(llm, retriever, st.session_state.chat_history, user_question)
                st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))

elif not st.session_state.chat_history:
    st.info("👆 Upload your BPCL FAQ PDF in the sidebar to start chatting.")
