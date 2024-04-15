import io
import os
import re
import shutil
import string
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from keras import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, TextVectorization

# Load your dataset
df = pd.read_csv('dataset/DGHD_mod.csv')

# Assuming 'text' is the column with text data and 'label' is the column with labels
texts = df['text'].values
labels = df['label'].values

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=123)

# Prepare a TensorFlow dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(1024)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(1024)

# Continue with the rest of the model preparation and training
vocab_size = 10000
sequence_length = 100

vectorize_layer = TextVectorization(
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)

# Adapt the vectorization layer to the dataset
text_ds = train_ds.map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

embedding_dim = 16

model = Sequential([
  vectorize_layer,
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=300)

weights = model.get_layer('embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
  if index == 0:
    continue  # skip 0, it's padding.
  vec = weights[index]
  out_v.write('\t'.join([str(x) for x in vec]) + "\n")
  out_m.write(word + "\n")
out_v.close()
out_m.close()
