import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import model
import threading
import tensorflow as tf
import nltk


nltk.download('stopwords')
nltk.download('wordnet')


# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

print_lock = threading.Lock()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def safe_print(message):
    with print_lock:
        print(message)


def load_dataset(file_path, sentence_col_idx, label_col_idx):
    data = pd.read_csv(file_path, encoding='UTF-8', header=None)

    # Extract relevant columns based on indices provided
    sentences = data.iloc[:, sentence_col_idx]
    labels = data.iloc[:, label_col_idx]

    return pd.DataFrame({'sentence': sentences, 'label': labels})


def preprocess(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)


data = load_dataset("dataset/spam_mod.csv", sentence_col_idx=1, label_col_idx=2)
data = data[pd.to_numeric(data['label'], errors='coerce').notna()]
data['label'] = data['label'].astype(int)
print(data.head())
data['sentence'] = data['sentence'].apply(preprocess)

# Tokenize and pad sequences
classes_num = 2
MAX_LEN = 35
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['sentence'])
X = tokenizer.texts_to_sequences(data['sentence'])
X = pad_sequences(X, maxlen=MAX_LEN)
y = to_categorical(data['label'])

# K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []

results = {}

for is_attention in [False, True]:
    for is_bidirectional in [False, True]:
        config_key = (is_attention, is_bidirectional)
        results[config_key] = []  # Initialize the list for this configuration

        for train, test in kfold.split(X, data['label']):
            # Instantiate model
            model_instance = model.lstm(vocab=tokenizer.word_index,
                                        hidden_units=128,
                                        num_layers=1,
                                        max_sequence_length=MAX_LEN,
                                        is_attention=is_attention,
                                        is_bidirectional=is_bidirectional,
                                        classes=classes_num)

            # Train model
            model_instance.fit(X[train], y[train], epochs=10, batch_size=32, verbose=1)

            # Evaluate model
            scores = model_instance.evaluate(X[test], y[test], verbose=0)
            results[config_key].append(scores[1] * 100)  # Append to the results list for this configuration
            safe_print(
                f"Model (Attention={is_attention},"
                f" Bidirectional={is_bidirectional}) - Accuracy: {scores[1] * 100:.2f}%")
        test_CP1 = True

# Determine and display the best result
best_config = max(results, key=lambda k: np.mean(results[k]))
best_acc = max(results[best_config])
best_mean = np.mean(results[best_config])
best_std = np.std(results[best_config])

safe_print(f"Best Model (Attention={best_config[0]}, Bidirectional={best_config[1]})")
safe_print(f"Accuracy: {best_acc:.2f}%")
safe_print(f"Mean accuracy for the run: {best_mean:.2f}%, Standard Deviation: {best_std:.2f}%")
