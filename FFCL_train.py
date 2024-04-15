import pandas as pd
import numpy as np
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.model_selection import StratifiedKFold, train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import model
import FFCL
import threading
import tensorflow as tf
import nltk
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api

from utils import *

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

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
    text = text.lower()
    text = ''.join([word for word in text if word not in string.punctuation])
    # tokens = text.split()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def visualize_attention(model_1, model_2, data, word_index, num_samples=5):
    # Predict and get attention weights
    predictions_1, attention_weights_1 = model_1.predict(data)
    predictions_2, attention_weights_2 = model_2.predict(data)

    # Select samples to visualize
    selected_indices = np.random.choice(data.shape[0], num_samples, replace=False)

    for i in selected_indices:
        sequence = data[i]
        # A/LSTM
        weights = attention_weights_1[i] / attention_weights_2[i]

        if len(weights.shape) == 2:
            weights = np.mean(weights, axis=-1)

        # Only consider non-zero parts of the sequence
        # word != 0
        words = [word for word in sequence if word != 0]
        words = [list(word_index.keys())[list(word_index.values()).index(w)] for w in words]
        attention = weights[:len(words)]

        # Plot
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(attention)), attention, align='center')
        plt.xticks(range(len(attention)), words, rotation=45)
        plt.xlabel('Words in Sequence')
        plt.ylabel('Attention Weight')
        plt.title(f'Sample {i} - Attention Mechanism Visualization')
        plt.show()


if __name__ == "__main__":
    data = load_dataset("dataset/DGHD_mod.csv.csv", sentence_col_idx=1, label_col_idx=2)
    data = data[pd.to_numeric(data['label'], errors='coerce').notna()]
    data['label'] = data['label'].astype(int)
    print(data.head())

    # data['sentence'] = data['sentence'].apply(preprocess)

    # Tokenize and pad sequences
    classes_num = max(data['label']) + 1
    MAX_LEN = 100
    tokenizer = Tokenizer()
    # 'sentence', 'text'
    tokenizer.fit_on_texts(data['sentence'])
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 100

    # X = tokenizer.texts_to_sequences(data['sentence'])
    # X = pad_sequences(X, maxlen=MAX_LEN)
    X = data['sentence'].apply(preprocess)
    y = to_categorical(data['label'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=data['label'],
                                                        shuffle=True,
                                                        random_state=42)

    corpus = api.load('glove-twitter-25')
    sentences = [sentence.split() for sentence in X]

    w2v_model = Word2Vec(sentences, window=5, min_count=5, workers=4)

    # Initialize the embedding matrix with zeros
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # Fill the embedding matrix with Word2Vec vectors
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_vector = w2v_model.wv[word]
            embedding_matrix[i] = embedding_vector


    def vectorize(sentence):
        words = sentence.split()
        words_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(words_vecs) == 0:
            return np.zeros(100)
        words_vecs = np.array(words_vecs)
        return words_vecs.mean(axis=0)

    # X_train = np.array([vectorize(sentence) for sentence in X_train])
    # X_test = np.array([vectorize(sentence) for sentence in X_test])

    # Convert sentences to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad sequences
    X_train_padded = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_padded = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    results = {}

    is_attention = True
    is_bidirectional = True
    is_lstm = False
    num_epoch = 17
    config_key = (is_attention, is_bidirectional)
    results[config_key] = []  # Initialize the list for this configuration

    config_key = (is_attention, is_bidirectional)
    results[config_key] = []  # Initialize the list for this configuration

    # Instantiate model
    model_instance = FFCL.cnn_lstm_parallel_with_cross_attention(vocab=tokenizer.word_index,
                                                                 embedding_matrix=embedding_matrix,
                                                                 embedding_dim=embedding_dim,
                                                                 hidden_units=256,
                                                                 num_layers=2,
                                                                 max_sequence_length=MAX_LEN,
                                                                 is_attention=is_attention,
                                                                 is_bidirectional=is_bidirectional,
                                                                 classes=classes_num,
                                                                 attention_con=False)

    # Train model
    model_instance.fit(X_train_padded, y_train, epochs=num_epoch, batch_size=64, verbose=1)

    if is_bidirectional and is_attention:
        attention_layer_connect = model_instance.get_layer(index=7)
        attention_layer_out = model_instance.get_layer(index=9)
        output_with_attention_connect = attention_layer_connect.output
        output_with_attention_out = attention_layer_out.output

        model_A_connect = Model(inputs=model_instance.input,
                                outputs=[model_instance.output, output_with_attention_connect])
        model_A_out = Model(inputs=model_instance.input,
                            outputs=[model_instance.output, output_with_attention_out])

    # Evaluate model
    scores = model_instance.evaluate(X_test_padded, y_test, verbose=0)
    results[config_key].append(scores[1] * 100)  # Append to the results list for this configuration
    safe_print(
        f"Model (Attention={is_attention},"
        f" Bidirectional={is_bidirectional}) - Accuracy: {scores[1] * 100:.2f}%")
    if is_bidirectional and is_attention and is_lstm:
        visualize_attention(model_A_out, model_A_connect, X_test, tokenizer.word_index, num_samples=5)

    # Determine and display the best result
    best_config = max(results, key=lambda k: results[k][0])
    best_acc = results[best_config][0]

    safe_print(f"Best Model (Attention={best_config[0]}, Bidirectional={best_config[1]})")
    safe_print(f"Accuracy: {best_acc:.2f}%")
