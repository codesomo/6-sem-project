import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import pandas as pd 
import numpy as np 
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import random

# Load pre-trained model and necessary data
model = load_model('model-v1.h5')
tokenizer_t = joblib.load('/home/somo/chatbot_project/retrival/Dumps/tokenizer_t.pkl')
vocab = joblib.load('/home/somo/chatbot_project/retrival/Dumps/vocab.pkl')
df2 = pd.read_csv('/home/somo/chatbot_project/retrival/response.csv')

lemmatizer = WordNetLemmatizer()

def tokenizer(entry):
    tokens = entry.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    tokens = [re_punc.sub('', w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [lemmatizer.lemmatize(w.lower()) for w in tokens]
    tokens = [word.lower() for word in tokens if len(word) > 1]
    return tokens

def remove_stop_words_for_input(tokenizer, df, feature):
    doc_without_stopwords = []
    for entry in df[feature]:
        tokens = tokenizer(entry)
        doc_without_stopwords.append(' '.join(tokens))
    df[feature] = doc_without_stopwords
    return df

def encode_input_text(tokenizer_t, df, feature):
    t = tokenizer_t
    entry = [df[feature][0]]
    encoded = t.texts_to_sequences(entry)
    padded = pad_sequences(encoded, maxlen=10, padding='post')
    return padded

def get_pred(model, encoded_input):
    pred = np.argmax(model.predict(encoded_input))
    return pred

def bot_precausion(df_input, pred):
    words = df_input.questions[0].split()
    if len([w for w in words if w in vocab]) == 0:
        pred = 1
    return pred

def get_response(df2, pred):
    upper_bound = df2.groupby('labels').get_group(pred).shape[0]
    r = np.random.randint(0, upper_bound)
    responses = list(df2.groupby('labels').get_group(pred).response)
    return responses[r]

def chatbot_response(user_input):
    df_input = pd.DataFrame([user_input], columns=['questions'])
    df_input = remove_stop_words_for_input(tokenizer, df_input, 'questions')
    encoded_input = encode_input_text(tokenizer_t, df_input, 'questions')
    pred = get_pred(model, encoded_input)
    pred = bot_precausion(df_input, pred)
    response = get_response(df2, pred)
    return response

if __name__ == "__main__":
    print("Bot: Hello! How can I help you with your mental health today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye! Take care!")
            break
        response = chatbot_response(user_input)
        print(f"Bot: {response}")
