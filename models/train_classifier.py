import sys
import pandas as pd
import numpy as np
import nltk

from joblib import dump
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    """
    Load and generate datasets for fitting along with message categories list
    
    Parameters
    -----------
    database_filepath : str
        SQLite database file path
    
    Returns
    ----------
    X : DataFrame
        Contains messages for generating features
    Y : DataFrame
        Contains binary labels for various message categories
    category_names : list
        List of different message categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("DisasterResponseData", con=engine)
    X = df["message"]
    Y = df[[col for col in df.columns.tolist() if col not in ["id", "message", "original", "genre"]]]
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Passed string is normalized, lemmatized, and tokenized
    
    Parameters
    -----------
    text : str
        text to be tokenized
    
    Returns
    ----------
    clean_tokens : list
        Contains generated tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    This transformer class extract the starting verb of a sentence
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Creates scikit Pipeline object for processing text messages and fitting a classifier.
    
    Parameters
    -----------
    None
    
    Returns
    ----------
    pipeline : Pipeline
        Pipeline object
    """
    pipeline = Pipeline([
                    ("features", FeatureUnion([
                        ('text_pipeline', Pipeline([
                            ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                            ('scaler', StandardScaler(with_mean=False))
                            ])),
                        ('tfidf_transformer', TfidfVectorizer()),
                        ('starting_verb_extr', StartingVerbExtractor())
                        ])),
                    ("clf", MultiOutputClassifier(RandomForestClassifier()))
                    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Method applies scikit pipeline to test set and prints the model performance (accuracy and f1score)
    
    Parameters
    -----------
    model : Pipeline
        fit pipeline
    X_test : ndarray
        test features
    Y_test : ndarray
        test labels
    category_names : list
        List of different message categories
    
    Returns
    ----------
    None
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save trained model
    
    Parameters
    -----------
    model : Pipeline
        fit pipeline
    model_filepath : str
        path with dump format
    
    Returns
    ----------
    None
    """
    dump(model, "{}".format(model_filepath))


def main():
    """
    Runner function
    
    This function:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()