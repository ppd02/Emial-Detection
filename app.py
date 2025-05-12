import streamlit as st
import pickle

# Load the saved model and vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Title and Description
st.title("📧 Email Spam Classifier")
st.write("Enter an email message below to check if it's **Spam** or **Ham**.")

# Input field
user_input = st.text_area("Enter Email Text", height=200)

# Prediction logic
if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Preprocessing function (same as training)
        import re
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer

        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()

        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'[^a-z\s]', '', text)
            tokens = text.split()
            tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
            return ' '.join(tokens)

        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed]).toarray()
        prediction = model.predict(vectorized)[0]

        # Display result
        if prediction == 1:
            st.error("🚫 This email is classified as **SPAM**.")
        else:
            st.success("✅ This email is classified as **HAM** (not spam).")
