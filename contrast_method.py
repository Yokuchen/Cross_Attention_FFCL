import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


class BernoulliNaiveBayes:
    def __init__(self):
        self.alpha = 1
        self.class_log_post = None
        self.feature_log_prob = None
        self.classes = None

    def fit(self, in_matrix, label):
        m, n = in_matrix.shape
        self.classes = np.unique(label)
        n_classes = len(self.classes)
        sample_class_count = 0
        for c in self.classes:
            if c in label:
                sample_class_count += 1
        # small bias to prevent 0 division
        self.class_log_post = np.log((np.array(sample_class_count) + 10e-7) / m)
        self.feature_log_prob = np.zeros((n_classes, n))

        for idx, c in enumerate(self.classes):
            x_c = in_matrix[label == c]
            # Laplace alpha(1)
            smoothed_count = 2 + x_c.shape[0]
            feature_counts = x_c.sum(axis=0) + 1
            self.feature_log_prob[idx] = np.log(feature_counts / smoothed_count)

    def predict(self, in_matrix):
        log_probs = self.predict_log_probability(in_matrix)
        return self.classes[np.argmax(log_probs, axis=1)]

    def predict_log_probability(self, in_matrix):
        return (in_matrix @ self.feature_log_prob.T) + self.class_log_post


# k-fold cross-validation
def cross_validate_model(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []

    for train_index, test_index in kf.split(X):
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        model.fit(X_train_k, y_train_k)
        predictions = model.predict(X_test_k)
        accuracy = accuracy_score(y_test_k, predictions)
        f1 = f1_score(y_test_k, predictions, average='weighted')
        accuracies.append(accuracy)
        f1_scores.append(f1)

    return np.mean(accuracies), np.mean(f1_scores)


def test_ngram(X, y, ngram_ranges):
    best_f1 = 0
    best_ngram = None
    for ngram_range in ngram_ranges:
        print(f"Testing with N-gram range: {ngram_range}")
        vectorizer = CountVectorizer(binary=True, max_features=3000, ngram_range=ngram_range)
        X_vectorized = vectorizer.fit_transform(X)

        _, f1_mean = cross_validate_model(BernoulliNaiveBayes(), X_vectorized, y)
        print(f"Mean F1 Score for N-gram {ngram_range}: {f1_mean}")

        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_ngram = ngram_range

    print(f"Best N-gram range: {best_ngram} with F1 Score: {best_f1}")
    return best_ngram


def preprocess(text):
    # Tokenize and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in word_tokenize(text)]
    # Remove stop words
    # if word not in stop_words
    return ' '.join([word for word in tokens if word not in stop_words])


if __name__ == "__main__":
    train_data = pd.read_csv('dataset/test_interview.csv')
    # preprocessing
    X = train_data['text'].apply(preprocess)
    y = train_data['label']
    # Testing different N-gram ranges
    ngram_ranges = [(1, 1), (1, 2), (1, 3)]
    best_ngram = test_ngram(X, y, ngram_ranges)
    # best_ngram = test_ngram(X, y, ngram_ranges)

    # Using the best N-gram range for final model training and prediction

    # Text Vectorization
    vectorizer = CountVectorizer(binary=True, max_features=3000)
    X_vectorized = vectorizer.fit_transform(X)

    # Splitting data
    X_train, X_val, y_train, y_val = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Training and evaluating
    bnb_model = BernoulliNaiveBayes()
    bnb_model.fit(X_train, y_train)
    y_pred = bnb_model.predict(X_val)
    accuracy_bnb = accuracy_score(y_val, y_pred)
    f1_bnb = f1_score(y_val, y_pred, average='weighted')
    recall_bnb = recall_score(y_val, y_pred)
    print(f"Bernoulli Naive Bayes\n Accuracy: {accuracy_bnb}, F1 Score: {f1_bnb}, Recall: {recall_bnb}")

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_val)
    accuracy_dt = accuracy_score(y_val, y_pred_dt)
    f1_dt = f1_score(y_val, y_pred_dt, average='weighted')
    recall_dt = recall_score(y_val, y_pred_dt)
    print(f"Decision Tree\n Accuracy: {accuracy_dt}, F1 Score: {f1_dt}, Recall: {recall_dt}")

    # Multi layer perceptron
    mlp_model = MLPClassifier(random_state=42, max_iter=300)
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_val)
    accuracy_mlp = accuracy_score(y_val, y_pred_mlp)
    f1_mlp = f1_score(y_val, y_pred_mlp, average='weighted')
    recall_mlp = recall_score(y_val, y_pred_mlp)
    print(f"Multi layer perceptron\n Accuracy: {accuracy_mlp}, F1 Score: {f1_mlp}, Recall: {recall_mlp}")

    cross_val = False
    if cross_val:
        # Cross-validation
        bnb_mean, bnb_f1_mean = cross_validate_model(BernoulliNaiveBayes(), X_vectorized, y)
        dt_mean, dt_f1_mean = cross_validate_model(DecisionTreeClassifier(random_state=42), X_vectorized, y)
        # mlp_mean, mlp_f1 = cross_validate_model(MLPClassifier(random_state=42, max_iter=500), X_vectorized, y)
        print(f"Bernoulli Naive Bayes - Cross validation\n Mean Accuracy: {bnb_mean}, Mean F1: {bnb_f1_mean}")
        print(f"Decision Tree - Cross validation\n Mean Accuracy: {dt_mean}, Mean F1: {dt_f1_mean}")
        # print(f"Multi layer perceptron - Cross validation\n Mean Accuracy: {mlp_mean}, Mean F1: {mlp_f1}")

    # Choosing the best model
    # models = [bnb_model, dt_model, mlp_model]
    # models_acc = [bnb_mean, dt_mean, mlp_mean]
    # best_model = bnb_model if bnb_mean > dt_mean else dt_model
    # if max(models_acc) is bnb_mean:
    #     model_name = "Bernoulli Naive Bayes"
    #     best_model = bnb_model
    # elif max(models_acc) is dt_model:
    #     model_name = "Decision Tree"
    #     best_model = dt_model
    # else:
    #     model_name = "Multi layer perceptron"
    #     best_model = mlp_model
    #
    # test_data = pd.read_csv('test.csv', encoding='ISO-8859-1')
    # X_test = vectorizer.transform(test_data['body'])
    #
    # test_predictions = best_model.predict(X_test)
    # test_data['subreddit'] = test_predictions
    #
    # # Saving predictions to a CSV file
    # output_df = test_data[['id', 'subreddit']]
    # output_df.to_csv('test_prediction.csv', index=False)
    # print("Predictions saved to 'test_prediction.csv'")
