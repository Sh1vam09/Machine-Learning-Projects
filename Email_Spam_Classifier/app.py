import streamlit as st
import re
import numpy as np
import pickle
import emoji
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from symspellpy.symspellpy import SymSpell, Verbosity
from nltk.stem import WordNetLemmatizer
import nltk
from gensim.models import Word2Vec
from numpy import triu
from collections import Counter
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up tools
stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
dictionary_path = os.path.join(BASE_DIR, "en-80k.txt")
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
model_path = os.path.join(BASE_DIR, "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)
w2v_path = os.path.join(BASE_DIR, "w2c.pkl")
with open(w2v_path, "rb") as f:
    w2v_model = pickle.load(f)
VECTOR_SIZE = 100  

# Preprocessing

def clean_email_text(text):
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ', text)
    text = re.sub(r'\b\d+[\).\]]', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', ' ', text)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b(re|fw):', ' ', text)
    text = re.sub(r'\b\w\b', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def average_word2vec(text,model,vector_size):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop]
    valid_words = [word for word in tokens if word in model.wv]
    if not valid_words:
        return np.zeros(vector_size)
    return np.mean([model.wv[word] for word in valid_words], axis=0)

def preprocessing(text):
    text = emoji.replace_emoji(text, replace='')
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop and word.isalpha()]
    corrected_tokens = []
    for word in tokens:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected_tokens.append(suggestions[0].term)
        else:
            corrected_tokens.append(word)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in corrected_tokens]
    return ' '.join(lemmatized_tokens)

def extract_text_features(text):
    num_chars = len(text)
    num_words = len(text.split())
    return [num_chars, num_words]

def repeated_word_features(text):
    words = text.lower().split()
    word_counts = Counter(words)
    repeated = [word for word, count in word_counts.items() if count > 1]
    return [len(repeated), len(repeated) / (len(words) + 1e-6)]

def predict(text):
    text=clean_email_text(text)
    vec =average_word2vec(text,w2v_model,VECTOR_SIZE)
    text_feats = extract_text_features(text)
    dup_count, dup_ratio = repeated_word_features(text)
    input_vector = np.hstack([vec, text_feats, dup_count, dup_ratio]).reshape(1, -1)
    label = model.predict(input_vector)[0]
    proba =model.predict_proba(input_vector)[0][1]
    return label, proba

# Streamlit UI
st.title("Email Spam Classifier")
user_input = st.text_area("Enter your email content here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, probability = predict(user_input)
        st.write("### Prediction:", "SPAM" if label == 1 else "NOT SPAM")
        st.write(f"### Spam Probability: `{probability:.2f}`")
# spam-env\Scripts\activate
