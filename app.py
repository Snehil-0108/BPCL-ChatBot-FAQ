import streamlit as st
import pandas as pd
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# ---------------------- INITIAL SETUP ----------------------

nltk.download('punkt')
nltk.download('stopwords')

# ---------------------- TEXT PREPROCESSING ----------------------

def preprocess_text(text):
"""Clean and tokenize text for vectorization."""
text = text.lower()
text = text.translate(str.maketrans('', '', string.punctuation))
tokens = word_tokenize(text)
tokens = [word for word in tokens if word not in stopwords.words('english')]
return " ".join(tokens)

# ---------------------- LOAD FAQ DATA ----------------------

@st.cache_data(show_spinner="📚 Loading FAQ knowledge base...")
def load_faq_data(file_path):
"""Load FAQ data from a text file."""
df = pd.read_csv(file_path, sep="|", header=None, names=["Question", "Answer"])
df["Cleaned_Q"] = df["Question"].apply(preprocess_text)
return df

# ---------------------- FIND BEST MATCH ----------------------

def get_best_answer(user_query, df):
"""Find the most relevant FAQ answer using cosine similarity."""
user_query_clean = preprocess_text(user_query)
corpus = df["Cleaned_Q"].tolist() + [user_query_clean]
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)
cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
index = cosine_sim.argmax()
score = cosine_sim[0][index]
if score < 0.2:
return "I'm sorry, I couldn’t find an exact match for your query. Please try rephrasing."
return df["Answer"].iloc[index]

# ---------------------- STREAMLIT APP ----------------------

st.set_page_config(page_title="BPCL NLP FAQ Chatbot", page_icon="🤖", layout="wide")

st.title("🧠 BPCL NLP FAQ Chatbot")
st.markdown("Ask me anything related to BPCL FAQs!")

faq_file = st.file_uploader("📤 Upload FAQ text file (use '|' separator)", type=["txt"])

if faq_file:
df = load_faq_data(faq_file)
st.success(f"✅ Loaded {len(df)} FAQs successfully!")
user_query = st.text_input("💬 Ask your question:")
if user_query:
with st.spinner("🤔 Thinking..."):
answer = get_best_answer(user_query, df)
st.write("**Answer:** ", answer)
else:
st.info("Please upload your FAQ text file to get started.")
