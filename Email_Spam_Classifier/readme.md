# 📧 Email Spam Classifier

A machine learning-powered web app that classifies emails as **Spam** or **Not Spam** using **custom NLP processing**, a **Word2Vec + XGBoost pipeline**, and **Optuna-based hyperparameter tuning** — all served via an interactive Streamlit interface.

---
**Dataset**
[Email_Spam_Balanced_Data](https://www.kaggle.com/datasets/shivam09baheti/email-spam-balanced-dataset)
## Features

- **Custom Text Preprocessing**:
  - Lowercasing
  - Punctuation and stopword removal
  - removed html tas,urls
  - chat word treatment
  - Lemmatization 
  - Spelling correction using `symspellpy` and `en-80k.txt`

- **Hand-Crafted Features**:
  - Number of Characters and Words
  - Repeated words

- **Custom-trained Word2Vec model on the dataset**
- **Average Word2Vec Embeddings**: Each email is converted into a single dense vector using the **mean of word vectors**
- **XGBoost Classifier** with **Optuna** hyperparameter tuning
- **Streamlit Web App** for real-time email prediction
- 🌐 Deployed on Streamlit Cloud
---
**Live Deployment**

The app is deployed on Streamlit Cloud.

[Email-Spam-Classifier](https://email-spam-classifier-swalla.streamlit.app/) 

