
# import pickle
# import nltk
# import speech_recognition as sr
# import re  # For regular expressions to count numbers, etc.

# # Download required NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

# # Load the trained model
# try:
#     with open("MNB.pkl", "rb") as f:
#         model = pickle.load(f)
#     with open("cv.pkl", "rb") as f:
#         cv = pickle.load(f)
# except FileNotFoundError:
#     print("❌ Error: 'MNB.pkl' or 'cv.pkl' not found. Please train the model and save it first.")
#     exit()

# # List of common spam keywords with assigned weights (higher weight = more suspicious)
# # This list can be dynamically updated based on analysis of the training data.
# spam_keyword_weights = {
#     "congratulations": 3, "won": 3, "lottery": 4, "prize": 4, 
#     "bank details": 5, "account number": 5, "click here": 4,
#     "free": 2, "urgent": 2, "win": 3, "cash": 3, "reward": 2, 
#     "gift": 2, "money": 3, "selected": 3, "claim": 4, "offer": 2, 
#     "transfer": 4, "verify": 3, "limited time": 2, "OTP": 5
# }

# # Text preprocessing and feature engineering function
# def preprocess_and_feature_engineer(text):
#     text = text.lower()
    
#     # 1. Numerical Features
#     num_digits = sum(c.isdigit() for c in text)
#     num_caps = sum(1 for c in text if c.isupper())
#     num_exclamations = text.count('!')
#     msg_length = len(text)
    
#     # 2. Text Preprocessing
#     lemmatizer = WordNetLemmatizer()
#     words = word_tokenize(text)
#     stop_words = set(stopwords.words('english'))
    
#     # Keep words that are alphanumeric or special spam-related words like "!"
#     words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
#     processed_text = " ".join(words)
    
#     return processed_text, num_digits, num_caps, num_exclamations, msg_length

# # Prediction function with hybrid keyword and ML model
# def predict_text(text):
#     processed_text, num_digits, num_caps, num_exclamations, msg_length = preprocess_and_feature_engineer(text)

#     # Calculate a score based on keyword presence
#     keyword_score = 0
#     for keyword, weight in spam_keyword_weights.items():
#         if keyword in processed_text:
#             keyword_score += weight
    
#     # Add to score based on numerical features
#     if num_digits > 3 or "OTP" in text:
#         keyword_score += 5
#     if num_exclamations > 2:
#         keyword_score += 2
#     if num_caps > 10 and num_caps / msg_length > 0.1: # if more than 10% of characters are caps
#         keyword_score += 3

#     # Use ML model as a second opinion
#     vectorized_text = cv.transform([processed_text]).toarray()
#     prediction_proba = model.predict_proba(vectorized_text)[0][1] # Probability of being spam
    
#     # Combine scores for a final, weighted decision
#     # You'll need to adjust these weights based on your model's performance.
#     # A simple example: 70% from ML model, 30% from keywords and features.
#     final_score = (prediction_proba * 0.7) + ((keyword_score / 20) * 0.3)
    
#     # Classify based on a threshold
#     return "Spam" if final_score > 0.5 or keyword_score > 10 else "Not Spam"

# # Speech recognition and prediction
# def main():
#     try:
#         recognizer = sr.Recognizer()
#         with sr.Microphone() as source:
#             print("🎤 Speak something (listening)...")
#             audio = recognizer.listen(source, timeout=5)

#         print("🔁 Recognizing speech...")
#         text = recognizer.recognize_google(audio)
#         print(f"📝 You said: {text}")

#         result = predict_text(text)
#         print(f"📊 Prediction: {result}")

#     except sr.WaitTimeoutError:
#         print("❌ Timeout: No speech detected")
#     except sr.UnknownValueError:
#         print("❌ Could not understand audio")
#     except sr.RequestError as e:
#         print(f"❌ API Error: {str(e)}")
#     except Exception as e:
#         print(f"❌ Unexpected Error: {str(e)}")

# # Run main
# if __name__ == "__main__":
#     main()

import joblib
import nltk
import speech_recognition as sr
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained SVM model and CountVectorizer
try:
    model = joblib.load("svm_model.pkl")
    cv = joblib.load("SVC_cv.pkl")
except FileNotFoundError:
    print("❌ Error: 'svm_model.pkl' or 'cv.pkl' not found. Train the model first.")
    exit()

# Optional: Spam/fraud keywords (can still be used for hybrid scoring)
spam_keyword_weights = {
    "congratulations": 3, "won": 3, "lottery": 4, "prize": 4, 
    "bank": 5, "account": 5, "click": 4,
    "free": 2, "urgent": 2, "win": 3, "cash": 3, "reward": 2, 
    "gift": 2, "money": 3, "selected": 3, "claim": 4, "offer": 2, 
    "transfer": 4, "verify": 3, "limited": 2, "otp": 5
}

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    
    # Tokenize and remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(w) for w in words if w.isalnum() and w not in stop_words]
    
    processed_text = " ".join(words)
    return processed_text

# Hybrid prediction function
def predict_text(text):
    processed_text = preprocess_text(text)
    
    # Keyword scoring
    keyword_score = 0
    for keyword, weight in spam_keyword_weights.items():
        if keyword in processed_text:
            keyword_score += weight
    
    # ML prediction
    vectorized_text = cv.transform([processed_text]).toarray()
    prediction_proba = model.predict_proba(vectorized_text)[0][1]  # probability of being fraud/spam
    
    # Combine scores (70% ML, 30% keyword heuristic)
    final_score = (prediction_proba * 0.7) + ((keyword_score / 20) * 0.3)
    
    return "Fraud/Spam" if final_score > 0.5 or keyword_score > 10 else "Normal"

# Speech recognition and prediction
def main():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Speak something (listening)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
        
        print("🔁 Recognizing speech...")
        text = recognizer.recognize_google(audio)
        print(f"📝 You said: {text}")
        
        result = predict_text(text)
        print(f"📊 Prediction: {result}")
    
    except sr.WaitTimeoutError:
        print("❌ Timeout: No speech detected")
    except sr.UnknownValueError:
        print("❌ Could not understand audio")
    except sr.RequestError as e:
        print(f"❌ API Error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected Error: {str(e)}")

# Run main
if __name__ == "__main__":
    main()
