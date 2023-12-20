import csv
import numpy as np
import seaborn as sns
import pandas as pd
from keras.models import Model
import math as math
import matplotlib.pyplot as plt


np.random.seed(0)
sns.set()


# Function to convert the CSV file
def convert_csv(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        # CSV reader and writer
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        writer.writerow(['id', 'text', 'label'])

        # Iterate over each row in the input file
        for id, row in enumerate(reader, start=1):
            # Extract label and text, ignoring extra commas
            if row[0] == 'ham':
                raw_label = 0
            elif row[0] == 'spam':
                raw_label = 1
            label, text = raw_label, row[1]
            # Write the new format [id, text, label]
            writer.writerow([id, text, label])


def convert_subreddit(input_file, output_file):
    # Label to integer mapping
    label_to_int = {'Toronto': 0, 'London': 1, 'Paris': 2, 'Montreal': 3}

    with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
            open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        # CSV reader and writer
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Skip the header of the input file
        next(reader)

        # Write the header to the output file
        writer.writerow(['id', 'text', 'label'])

        # Iterate over each row in the input file
        for id, row in enumerate(reader, start=1):
            # Extract body and subreddit, convert label to integer
            body, subreddit = row[0], row[1]
            int_label = label_to_int[subreddit]
            # Write the new format [id, body, int_label]
            writer.writerow([id, body, int_label])


def show_attention(sentences, intermediate_layer_model1, intermediate_layer_model2, model):
    fig, axn = plt.subplots(len(sentences), 1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=5, hspace=10)
    fig.tight_layout()

    for i, ax in enumerate(axn.flat):
        seq = sentences[i]
        words = seq.split(" ")
        arr = np.zeros(35)
        word_to_idx = dict()
        current_key = 1
        for word in words:
            word = word.lower()
            if not (word in word_to_idx):
                word_to_idx[word] = current_key
                current_key += 1

        for j in range(len(words)):
            if words[j] in word_to_idx:
                arr[j] = word_to_idx[words[j].lower()]
            else:
                arr[j] = word_to_idx[""]

        arr = np.reshape(arr, (1, arr.shape[0]))
        intermediate_output2 = intermediate_layer_model2.predict(arr, verbose=0)
        intermediate_output1 = intermediate_layer_model1.predict(arr, verbose=0)
        print(seq, model.predict(arr))

        weights = intermediate_output2 / intermediate_output1
        val = []
        total = 0
        for j in range(len(words)):
            val.append(weights[0][j][0])
            total += weights[0][j][0]

        d = {}
        print(val)
        d[""] = pd.Series(val, index=words)

        df = pd.DataFrame(d)
        df.reindex(sentences[i].split(" "))
        df = df.transpose()

        sns.heatmap(df, ax=ax, annot=False, cbar_ax=None, cbar=False, linewidths=.0,
                    cmap="YlGnBu")  # "Oranges")#"RdBu_r")

    return fig
