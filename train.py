import pandas as pd
import numpy as np
import re
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv(r"D:\SEMESTER 6\DEEP LEARNING\Processed Dataset.csv")

print("Data shape:", df.shape)
print(df.head())

# =========================
# 2. CEK KOLOM WAJIB
# =========================
print("Kolom dataset:", df.columns)

# =========================
# 3. CLEANING TEXT SEDERHANA
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

df['text'] = df['reviewContent'].apply(clean_text)

# =========================
# 4. LABEL ENCODING
# =========================
le = LabelEncoder()
df['label'] = le.fit_transform(df['sentiment_label'])

print("Kelas:", le.classes_)

# =========================
# 5. SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['label'],
    test_size=0.2,
    random_state=42
)

# =========================
# 6. TOKENIZER
# =========================
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# =========================
# 7. PADDING
# =========================
max_len = 100

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# =========================
# 8. MODEL LSTM
# =========================
model = Sequential()

model.add(Embedding(5000, 128, input_length=max_len))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =========================
# 9. TRAINING MODEL
# =========================
history = model.fit(
    X_train_pad,
    y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test_pad, y_test)
)

# =========================
# 10. SIMPAN MODEL
# =========================
os.makedirs("model", exist_ok=True)

model.save("model/sentiment_lstm.h5")

pickle.dump(tokenizer, open("model/tokenizer.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print("Model berhasil disimpan di folder model/")