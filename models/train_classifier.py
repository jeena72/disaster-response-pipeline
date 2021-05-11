import sys
import pandas as pd
import numpy as np
import nltk
from joblib import dump
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


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
    pipeline = Pipeline([("features",
                      TfidfVectorizer(tokenizer=tokenize)),
                    ("clf", MultiOutputClassifier(LogisticRegression()))])
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