import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import model
import threading
import tensorflow as tf
import nltk

from utils import *

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
    # tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
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


data = load_dataset("dataset/subreddit_mod.csv", sentence_col_idx=1, label_col_idx=2)
data = data[pd.to_numeric(data['label'], errors='coerce').notna()]
data['label'] = data['label'].astype(int)
print(data.head())
data['sentence'] = data['sentence'].apply(preprocess)

# Tokenize and pad sequences
classes_num = max(data['label']) + 1
MAX_LEN = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['sentence'])
X = tokenizer.texts_to_sequences(data['sentence'])
X = pad_sequences(X, maxlen=MAX_LEN)
y = to_categorical(data['label'])

# K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)
cvscores = []

results = {}
subreddit_test = ["this is a data should not care created at Mcgill in montreal at quebec in Canada",
                  "this part of the text does not matter fo the test, but Toronto in Ontario near York matter",
                  "get a new iphone at apple store and it is cheap at this location in London"]
subreddit_test = tokenizer.texts_to_sequences(subreddit_test)

for is_attention in [False, True]:
    for is_bidirectional in [False, True]:
        config_key = (is_attention, is_bidirectional)
        results[config_key] = []  # Initialize the list for this configuration

        for train, test in kfold.split(X, data['label']):
            # Instantiate model
            model_instance = model.lstm(vocab=tokenizer.word_index,
                                        hidden_units=512,
                                        num_layers=2,
                                        max_sequence_length=MAX_LEN,
                                        is_attention=is_attention,
                                        is_bidirectional=is_bidirectional,
                                        classes=classes_num,
                                        attention_con=False)

            # Train model
            model_instance.fit(X[train], y[train], epochs=10, batch_size=32, verbose=1)

            if is_bidirectional and is_attention:
                attention_layer_connect = model_instance.get_layer(index=2)
                attention_layer_out = model_instance.get_layer(index=3)
                output_with_attention_connect = attention_layer_connect.output
                output_with_attention_out = attention_layer_out.output

                model_A_connect = Model(inputs=model_instance.input,
                                        outputs=[model_instance.output, output_with_attention_connect])
                model_A_out = Model(inputs=model_instance.input,
                                    outputs=[model_instance.output, output_with_attention_out])

            # Evaluate model
            scores = model_instance.evaluate(X[test], y[test], verbose=0)
            results[config_key].append(scores[1] * 100)  # Append to the results list for this configuration
            safe_print(
                f"Model (Attention={is_attention},"
                f" Bidirectional={is_bidirectional}) - Accuracy: {scores[1] * 100:.2f}%")
            if is_bidirectional and is_attention:
                # visualize_attention(model_A_connect, X[test], tokenizer.word_index, num_samples=5)
                visualize_attention(model_A_out, model_A_connect, X[test], tokenizer.word_index, num_samples=5)
                # show_attention(stressed, model_A_connect, model_A_out, model_instance)
        test_CP1 = True


# Determine and display the best result
best_config = max(results, key=lambda k: np.mean(results[k]))
best_acc = max(results[best_config])
best_mean = np.mean(results[best_config])
best_std = np.std(results[best_config])

safe_print(f"Best Model (Attention={best_config[0]}, Bidirectional={best_config[1]})")
safe_print(f"Accuracy: {best_acc:.2f}%")
safe_print(f"Mean accuracy for the run: {best_mean:.2f}%, Standard Deviation: {best_std:.2f}%")
