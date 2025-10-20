import os
import streamlit as st
import pandas as pd
import numpy as np
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#---------------------- INITIAL SETUP ----------------------



nltk.download('punkt')
nltk.download('stopwords')

#---------------------- TEXT PREPROCESSING ----------------------



def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

#---------------------- LOAD FAQ DATA ----------------------



@st.cache_data(show_spinner="📚 Loading FAQ knowledge base...")
def load_faq_data(file_path="faq_data.txt"):
        df = pd.read_csv(file_path, sep="|", header=None, names=["Question", "Answer"])
        df["Cleaned_Q"] = df["Question"].apply(preprocess_text)
        return df

#---------------------- GET RESPONSE ----------------------



def get_response(user_input, faq_df):
    query_clean = preprocess_text(user_input)
    corpus = faq_df["Cleaned_Q"].tolist() + [query_clean]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf[-1], tfidf[:-1])
    idx = np.argmax(similarity)
    if similarity[0][idx] > 0.3:
        return faq_df.iloc[idx]["Answer"]
    else:
        return "Sorry, I’m not sure about that. Please contact BPCL support for assistance."

#---------------------- STREAMLIT APP ----------------------



st.set_page_config(page_title="🧠 BPCL FAQ Chatbot", page_icon="🤖", layout="wide")




if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faq_df" not in st.session_state:
    st.session_state.faq_df = None

#---------------------- SIDEBAR ----------------------



with st.sidebar:
    st.title("⚙️ Settings")
st.markdown("Upload your BPCL FAQ dataset or use the default one.")

uploaded_file = st.file_uploader("📤 Upload a FAQ text file", type=["txt"])
if uploaded_file:
    with open("faq_data.txt", "wb") as f:
        f.write(uploaded_file.read())
    st.session_state.faq_df = load_faq_data("faq_data.txt")
    st.success("✅ Custom FAQ file loaded!")
else:
    st.session_state.faq_df = load_faq_data()

if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

#---------------------- MAIN CHAT INTERFACE ----------------------



st.title("🤖 BPCL NLP FAQ Chatbot")
st.markdown("Ask questions related to BPCL — the bot will respond based on stored FAQs.")

#Display chat history



for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

#Chat input



if user_question := st.chat_input("Ask a BPCL-related question..."):
    st.session_state.chat_history.append(("user", user_question))
    with st.chat_message("user"):
        st.markdown(user_question)

with st.chat_message("assistant"):
    with st.spinner("🧠 Thinking..."):
        answer = get_response(user_question, st.session_state.faq_df)
        st.markdown(answer)
st.session_state.chat_history.append(("assistant", answer))

