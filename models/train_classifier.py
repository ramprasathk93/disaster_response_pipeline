import sys
import re
import warnings
import nltk
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.externals import joblib

nltk.download(['punkt', 'stopwords', 'wordnet'])
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Load the messages_categories table from the SQLite database file

    :param database_filepath: SQLite database filepath
    :return: X: dataframe with all the disaster messages
    :return: Y: dataframe with all the respective classification category
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', con=engine)
    X = df["message"]
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    Y['related'] = Y['related'].map(lambda x: 1 if x == 2 else x)
    return X, Y


def tokenize(text):
    """
    Tokenize the given text with the following steps:
    - Pick only alphanumeric text and make everything lowercase
    - Lemmatise all the words as a verb
    - Remove the stopwords

    :param text: set of words to be used in training the classifier
    :return: set of tokens for training the model
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # normalize
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)  # tokenize
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in words]
    stop_words_set = stopwords.words("english")  # load stop words
    words = [word for word in tokens if word not in stop_words_set]  # remove stop words
    return words


def build_model():
    """
    Build a pipeline model and use grid search for tuning the hyper parameters
    Pipeline uses the following classes in the sequence:
    1. CountVectorizer
    2. Tf-Idf transformer
    3. Random Forest Classifier

    :return: Grid search model ready to fit
    """
    pipeline = Pipeline([
        ("count", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3, 4],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=3)
    return model


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the fitted model with the testing data.
    Uses the classification report from sklearn for evaluating each message category

    :param model: Fitted model
    :param X_test: Testing features
    :param Y_test: Testing labels

    """
    Y_pred = model.predict(X_test)
    print('Overall Accuracy: {}'.format(np.mean(Y_test.values == Y_pred)))
    print('Printing classification report')
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the trained model in a pickle file using sklearn's joblib

    :param model: Trained model
    :param model_filepath: Path to save the pickle file
    """
    # https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
