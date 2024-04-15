import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers.schedules import ExponentialDecay

from keras.layers import Dense, Dropout, Activation, Reshape, Flatten, LSTM, Dense, Dropout, Embedding, Bidirectional, \
    GRU, Conv1D, MaxPooling1D, Input, Concatenate, Input, Concatenate, Permute, Dot, Softmax, Multiply, Add, Lambda
from keras import initializers, regularizers
from keras import optimizers
from keras.layers import Layer
from keras import constraints
import tensorflow as tf


def cnn_lstm(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional, classes, attention_con=False):
    timesteps = max_sequence_length
    num_classes = classes

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=10000,
        decay_rate=0.9)
    # adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    adam = optimizers.Adam(learning_rate=lr_schedule)

    model = Sequential()
    model.add(Embedding(len(vocab)+1, 300, input_length=100))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    for i in range(num_layers):
        return_sequences = is_attention or (num_layers > 1 and i < num_layers - 1)

        if is_bidirectional:
            model.add(Bidirectional(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                                         kernel_initializer=initializers.glorot_normal(seed=777),
                                         bias_initializer='zeros')))
            if attention_con:
                model.add(AttentionWithContext())
                model.add(Bidirectional(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                                             kernel_initializer=initializers.glorot_normal(seed=777),
                                             bias_initializer='zeros')))
        else:
            model.add(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                           kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'))
            if attention_con:
                model.add(AttentionWithContext())
                model.add(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                               kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'))

    if is_attention:
        model.add(AttentionWithContext())
        model.add(Addition())

    model.add(Dense(num_classes, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=777),
                    bias_initializer='zeros'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    model.summary()

    return model


def cnn_lstm_parallel(vocab, hidden_units, num_layers, max_sequence_length, is_attention, is_bidirectional, classes, attention_con=False):
    # CNN branch
    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(len(vocab)+1, 300, input_length=100)(input_layer)

    cnn = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedding_layer)
    cnn = MaxPooling1D(pool_size=2)(cnn)

    cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Flatten()(cnn)

    # LSTM branch
    lstm_input = embedding_layer  # Using the same embedding layer as input
    for i in range(num_layers):
        return_sequences = is_attention or (num_layers > 1 and i < num_layers - 1)
        if is_bidirectional:
            lstm = Bidirectional(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                                      kernel_initializer=initializers.glorot_normal(seed=777),
                                      bias_initializer='zeros'))(lstm_input)
        else:
            lstm = LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                        kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros')(lstm_input)
        if i == num_layers - 1:
            # Only applying attention after the last layer
            lstm = AttentionWithContext()(lstm)
            lstm = Flatten()(lstm)

    # Concatenate CNN and LSTM outputs
    concatenated = Concatenate()([cnn, lstm])

    # Final dense layer
    output_layer = Dense(classes, activation='softmax', kernel_initializer=initializers.glorot_normal(seed=777),
                         bias_initializer='zeros')(concatenated)

    model = Model(inputs=input_layer, outputs=output_layer)
    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=10000,
        decay_rate=0.9)
    # adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    adam = optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])
    model.summary()

    return model


def cross_attention(query, value):
    attention_scores = Dot(axes=[2, 2])([query, value])
    attention_weights = Softmax(axis=-1)(attention_scores)

    # Expand dimensions of attention_weights to match value for element-wise multiplication
    attention_weights_expanded = K.expand_dims(attention_weights, axis=-1)

    # Ensure value is ready for multiplication
    value_expanded = K.expand_dims(value, axis=1)

    weighted_sum = Multiply()([attention_weights_expanded, value_expanded])

    # Sum over the attention axis to collapse the weighted features back to original tensor shape
    weighted_sum = K.sum(weighted_sum, axis=2)

    return weighted_sum


def cosine_similarity(query, value):
    """
    Compute cosine similarity
    Assumes query and value tensors have last dimension as features
    """
    # Ensure the last dimension size matches between query and value
    query_norm = K.l2_normalize(query, axis=-1)
    value_norm = K.l2_normalize(value, axis=-1)

    cos_similarity = K.sum(query_norm * value_norm, axis=-1, keepdims=True)
    return cos_similarity


def cross_attention_with_cosine_similarity(query, value):
    """
    Cross-attention using cosine similarity
    Assumes query and value are of compatible shapes and the last dimension is features
    """
    attention_scores = Lambda(lambda x: cosine_similarity(x[0], x[1]))([query, value])

    # Apply softmax to obtain attention weights
    attention_weights = Softmax(axis=-1)(attention_scores)

    # Expanding dimensions of value for multiplication to match attention weights shape
    value_expanded = K.expand_dims(value, axis=-2)
    attention_weights_expanded = K.expand_dims(attention_weights, axis=-1)

    # Weighted sum of value vectors
    weighted_sum = Multiply()([attention_weights_expanded, value_expanded])
    # Summing over the attention axis to collapse it
    weighted_sum = K.sum(weighted_sum, axis=-2)

    return weighted_sum


def cnn_lstm_parallel_with_cross_attention(vocab, embedding_matrix, embedding_dim, hidden_units, num_layers,
                                           max_sequence_length, is_attention, is_bidirectional, classes,
                                           attention_con=False):
    # embedding_dim = 300/100
    input_layer = Input(shape=(max_sequence_length,))
    # embedding_layer = Embedding(len(vocab)+1, 300, input_length=max_sequence_length)(input_layer)
    if embedding_matrix is not None:
        embedding_layer = Embedding(input_dim=len(vocab)+1,
                                    output_dim=embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=max_sequence_length,
                                    trainable=True)(input_layer)
    else:
        embedding_layer = Embedding(len(vocab) + 1, embedding_dim, input_length=max_sequence_length)(input_layer)

    # CNN branch
    cnn = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedding_layer)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)
    cnn = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(cnn)
    cnn = MaxPooling1D(pool_size=2)(cnn)

    # LSTM branch
    lstm_input = embedding_layer
    for i in range(num_layers):
        return_sequences = is_attention or (num_layers > 1 and i < num_layers - 1)
        if is_bidirectional:
            lstm = Bidirectional(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2))(lstm_input)
        else:
            lstm = LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2)(lstm_input)
        if i == num_layers - 1 and is_attention:  # Assume AttentionWithContext() is defined elsewhere
            lstm = AttentionWithContext()(lstm)

    # Project LSTM output to match CNN's feature dimension
    lstm_projected = Dense(128, activation='relu')(lstm)

    # Cross-Attention
    cnn_attended_to_lstm = cross_attention(cnn, lstm_projected)
    lstm_attended_to_cnn = cross_attention(lstm_projected, cnn)
    # Concatenate and Flatten
    combined_feature = Concatenate()([Flatten()(cnn_attended_to_lstm), Flatten()(lstm_attended_to_cnn)])

    output_layer = Dense(classes, activation='softmax')(combined_feature)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
    model.summary()

    return model


##############################################
"""
# ATTENTION LAYER

Using a context vector to assist the attention

* How to use:
Put return_sequences=True on the top of an RNN Layer (GRU/LSTM/SimpleRNN).
The dimensions are inferred based on the output shape of the RNN.

Example:
    model.add(LSTM(64, return_sequences=True))
    model.add(AttentionWithContext())
    model.add(Addition())
    # next add a Dense layer (for classification/regression) or whatever...
"""


##############################################
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.

    follows these equations:

    (1) u_t = tanh(W h_t + b)
    (2) \alpha_t = \frac{exp(u^T u)}{\sum_t(exp(u_t^T u))}, this is the attention weight
    (3) v_t = \alpha_t * h_t, v in time t

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, steps, features)`.

    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero and this results in NaN's.
        # Should add a small epsilon as the workaround
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        return weighted_input

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2]


class Addition(Layer):
    """
    This layer is supposed to add of all activation weight.
    We split this from AttentionWithContext to help us getting the activation weights

    follows this equation:

    (1) v = \sum_t(\alpha_t * h_t)

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim


if __name__ == "__main__":
    import numpy as np
    from keras.utils import to_categorical

    def generate_mockup_data(num_samples, vocab_size, max_sequence_length):
        # Randomly generate sequences
        X = np.random.randint(vocab_size, size=(num_samples, max_sequence_length))
        # Randomly generate labels (binary classification)
        y = np.random.randint(2, size=num_samples)
        return X, to_categorical(y)


    # Constants
    VOCAB_SIZE = 10000
    MAX_SEQUENCE_LENGTH = 50
    NUM_SAMPLES = 1000
    HIDDEN_UNITS = 256
    NUM_LAYERS = 1
    embedding_dim = 100
    IS_ATTENTION = True
    IS_BIDIRECTIONAL = True

    # Generate training data and test data
    X_train, y_train = generate_mockup_data(NUM_SAMPLES, VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    X_test, y_test = generate_mockup_data(NUM_SAMPLES, VOCAB_SIZE, MAX_SEQUENCE_LENGTH)
    emb_mat = []

    embedding_matrix = np.zeros((VOCAB_SIZE, embedding_dim))
    model = cnn_lstm_parallel_with_cross_attention(range(VOCAB_SIZE-1), embedding_matrix, 100, HIDDEN_UNITS,
                                                   NUM_LAYERS, MAX_SEQUENCE_LENGTH,
                                                   IS_ATTENTION, IS_BIDIRECTIONAL, 2)

    # Train
    model.fit(X_train, y_train, batch_size=32, epochs=5, validation_data=(X_test, y_test))

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, batch_size=32)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
