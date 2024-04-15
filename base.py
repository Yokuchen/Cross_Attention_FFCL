import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import Model
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import model
import FFCL
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


def debias_sentences(sentences, words_to_debias, special_token):
    """
    Replaces specified words in the sentences with a special token.

    :param sentences: List of sentences (strings) to be processed.
    :param words_to_debias: List of words to replace with the special token.
    :param special_token: The token used to replace words.
    :return: List of processed sentences with specific words replaced by the special token.
    """
    debiased_sentences = []
    for sentence in sentences:
        debiased_sentence = sentence
        for word in words_to_debias:
            # Replace the word with the special token
            debiased_sentence = debiased_sentence.replace(word, special_token)
        debiased_sentences.append(debiased_sentence)
    return debiased_sentences


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


def visualize_attention_indices(model_1, model_2, data, word_index, samples, mode='sentences'):
    """
    Visualizes the attention mechanism for selected samples or sentences.

    :param model_1: The first model for prediction.
    :param model_2: The second model for prediction.
    :param data: The dataset containing sequences.
    :param word_index: Mapping of words to their indices in the tokenizer.
    :param samples: List of indices or sentences to visualize.
    :param mode: 'indices' for selecting samples by indices, 'sentences' for processing a list of sentences.
    """
    selected_sequences = []
    if mode == 'indices':
        # Select samples directly by indices
        selected_indices = samples[:4]
        selected_sequences = data[selected_indices]
    elif mode == 'sentences':
        # Ensure samples are strings and preprocess
        assert all(isinstance(sample, str) for sample in samples), "All samples must be strings in 'sentences' mode."
        processed_sentences = [preprocess(sentence) for sentence in samples]
        sequences = tokenizer.texts_to_sequences(processed_sentences)
        selected_sequences = pad_sequences(sequences, maxlen=MAX_LEN)
        selected_indices = range(len(selected_sequences))  # For labeling in visualization
    else:
        raise ValueError("Invalid mode. Choose 'indices' or 'sentences'.")

    for i, sequence in zip(selected_indices, selected_sequences):
        # Predict and get attention weights
        predictions_1, attention_weights_1 = model_1.predict(np.array([sequence]))
        predictions_2, attention_weights_2 = model_2.predict(np.array([sequence]))

        weights = attention_weights_1/attention_weights_2
        if len(weights.shape) == 2:
            weights = np.mean(weights, axis=-1)

        words = [word for word in sequence if word != 0]
        words = [list(word_index.keys())[list(word_index.values()).index(w)] for w in words]
        attention = weights[0, :len(words)]  # Adjust indexing based on output shape
        attention_avg = np.mean(attention, axis=-1)
        sequence_length = len([word for word in sequence if word != 0])

        # Plot
        plt.figure(figsize=(10, 5))
        plt.bar(range(sequence_length), attention_avg, align='center')
        plt.xticks(range(sequence_length), words, rotation=45)
        plt.xlabel('Words in Sequence')
        plt.ylabel('Attention Weight')
        plt.title(f'Sample {i} - Attention Mechanism Visualization')
        plt.show()


def print_confusion_details(cm):
    # cm is a confusion matrix: rows are actual, columns are predicted
    class_count = cm.shape[0]  # Number of classes

    for cls in range(class_count):
        TP = cm[cls, cls]
        FP = cm[:, cls].sum() - TP  # Sum of the column for cls, minus TP
        FN = cm[cls, :].sum() - TP  # Sum of the row for cls, minus TP
        TN = cm.sum() - (FP + FN + TP)

        print(f"Class {cls} Metrics:")
        print(f"  True Positives (TP): {TP}")
        print(f"  True Negatives (TN): {TN}")
        print(f"  False Positives (FP): {FP}")
        print(f"  False Negatives (FN): {FN}")
        print("")


if __name__ == "__main__":
    male_words = [
        "father", "son", "brother", "uncle", "grandfather", "man", "gentleman",
        "boy", "male", "husband", "boyfriend", "nephew", "he", "him", "his"
    ]

    female_words = [
        "mother", "daughter", "sister", "aunt", "grandmother", "woman", "lady",
        "girl", "female", "wife", "girlfriend", "niece", "she", "her", "hers"
    ]
    words_to_debias = male_words + female_words
    special_token = '<UNK>'

    data = load_dataset("dataset/filtered_DGHD_mod.csv", sentence_col_idx=1, label_col_idx=2)
    data = data[pd.to_numeric(data['label'], errors='coerce').notna()]
    data['label'] = data['label'].astype(int)
    print(data.head())
    data['sentence'] = debias_sentences(data['sentence'].tolist(), words_to_debias, special_token)
    data['sentence'] = data['sentence'].apply(preprocess)

    # Tokenize and pad sequences
    classes_num = max(data['label']) + 1
    binary = True if classes_num == 2 else False
    MAX_LEN = 50
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['sentence'])
    X = tokenizer.texts_to_sequences(data['sentence'])
    X = pad_sequences(X, maxlen=MAX_LEN)
    y = to_categorical(data['label'])

    # K-Fold Cross-Validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True)
    cvscores = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        shuffle=False, random_state=42)

    results = {}
    subreddit_test = ["she can not join this university as its for intelligent people and women are not intelligent.",
                      "she can not join this university as its for intelligent people and she is not that intelligent.",
                      "My mate was telling me about this beautiful woman he was talking to. She was absolutely stunning but it turned out she was cheating on him, what a bitch",
                      "i'm glad i never asked out and nobody should ever do that. She is so selfish and privileged it is unbearable. Never marry a lady like that. Or anyone like that.",
                      "violence against women and victim blaming are really not cool, even if she thinks differently to you. Calling her a slut either"]
    # subreddit_test = tokenizer.texts_to_sequences(subreddit_test)

    is_attention = True
    is_bidirectional = True
    is_lstm = False
    num_epoch = 5
    config_key = (is_attention, is_bidirectional)
    results[config_key] = []  # Initialize the list for this configuration

    config_key = (is_attention, is_bidirectional)
    results[config_key] = []  # Initialize the list for this configuration

    # Instantiate model
    model_instance = model.lstm(vocab=tokenizer.word_index,
                                hidden_units=256,
                                num_layers=1,
                                max_sequence_length=MAX_LEN,
                                is_attention=is_attention,
                                is_bidirectional=is_bidirectional,
                                classes=classes_num,
                                attention_con=False)

    # Train model
    model_instance.fit(X_train, y_train, epochs=num_epoch, batch_size=32, verbose=1)

    if is_bidirectional and is_attention:
        attention_layer_connect = model_instance.get_layer(index=1)
        attention_layer_out = model_instance.get_layer(index=2)
        output_with_attention_connect = attention_layer_connect.output
        output_with_attention_out = attention_layer_out.output

        model_A_connect = Model(inputs=model_instance.input,
                                outputs=[model_instance.output, output_with_attention_connect])
        model_A_out = Model(inputs=model_instance.input,
                            outputs=[model_instance.output, output_with_attention_out])

    # Evaluate model
    scores = model_instance.evaluate(X_test, y_test, verbose=0)
    results[config_key].append(scores[1] * 100)  # Append to the results list for this configuration
    safe_print(
        f"Model (Attention={is_attention},"
        f" Bidirectional={is_bidirectional}) - Accuracy: {scores[1] * 100:.2f}%")

    # if is_bidirectional and is_attention:
    #     visualize_attention_indices(model_A_connect, model_A_out, X_test, tokenizer.word_index, subreddit_test)

    predictions_proba = model_instance.predict(X_test)
    predictions = np.argmax(predictions_proba, axis=1)
    true_labels = np.argmax(y_test, axis=1)

    # Identify misclassified instances
    misclassified_indices = np.where(predictions != true_labels)[0]

    num_samples = min(10, len(misclassified_indices))
    selected_misclassified_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
    manual_select = [811]

    # print(X_test[selected_misclassified_indices])
    # print(selected_misclassified_indices)
    if is_bidirectional and is_attention:
        # sentences indices
        visualize_attention_indices(model_A_connect, model_A_out, X_test,
                                    tokenizer.word_index, subreddit_test, mode='sentences')

    # Determine and display the best result
    best_config = max(results, key=lambda k: results[k][0])
    best_acc = results[best_config][0]

    safe_print(f"Best Model (Attention={best_config[0]}, Bidirectional={best_config[1]})")
    safe_print(f"Accuracy: {best_acc:.2f}%")
    cm = confusion_matrix(true_labels, predictions)
    if binary:
        TN, FP, FN, TP = cm.ravel()

        print(f"True Positives: {TP}")
        print(f"True Negatives: {TN}")
        print(f"False Positives: {FP}")
        print(f"False Negatives: {FN}")
    else:
        print_confusion_details(cm)
