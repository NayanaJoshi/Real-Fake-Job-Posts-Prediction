import streamlit as st 
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')  # For multilingual WordNet support

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    """Cleans and preprocesses text for model prediction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = [word for word in text.split() if word not in stop_words]  # Remove stopwords
    tokens = nltk.word_tokenize(" ".join(words))  # Tokenize
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]  # Lemmatize
    return " ".join(processed_tokens)

# Load the vectorizer and model with error handling
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("The vectorizer.pkl file is missing. Please ensure it's in the same directory.")
    st.stop()

try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    st.error("The model.pkl file is missing. Please ensure it's in the same directory.")
    st.stop()

# Streamlit app interface
st.title("Real/Fake Job Post Detector")

# Input fields for user
st.header("Input the Job Posting Details Below:")
job_title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

# Prediction button
if st.button("Predict"):
    # Preprocess and combine inputs
    preprocessed_title = preprocess_text(job_title)
    preprocessed_company_profile = preprocess_text(company_profile)
    preprocessed_description = preprocess_text(description)
    preprocessed_requirements = preprocess_text(requirements)
    preprocessed_benefits = preprocess_text(benefits)
    
    combined_text = f"{preprocessed_title} {preprocessed_company_profile} {preprocessed_description} {preprocessed_requirements} {preprocessed_benefits}"

    # Vectorize the input
    try:
        vector_input = tfidf.transform([combined_text])
    except Exception as e:
        st.error(f"Error during vectorization: {e}")
        st.stop()

    # Predict using the model
    try:
        prediction = model.predict(vector_input)[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.stop()

    # Display the result
    if prediction == 1:
        st.header("⚠️ Detected as Fake")
        st.caption("This job post seems suspicious.")
    else:
        st.header("✅ Detected as Real")
        st.caption("This job post appears to be legitimate.")
