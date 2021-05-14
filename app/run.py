import json
import plotly
import pandas as pd
import joblib
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseData', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # for plot-1
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)
    # for plot-2
    labels_data = df[[col for col in df.columns.tolist() if col not in ["id", "message", "original", "genre"]]]
    imbalance_df = pd.concat([pd.Series(labels_data.mean(), name="1"), pd.Series(1 - labels_data.mean(), name="0")], axis=1)
    imbalance_df.sort_values(by=["1"], inplace=True)
    message_categories_list = imbalance_df.index.tolist()
    ones_count_normalized = imbalance_df["1"]
    zeros_count_normalized = imbalance_df["0"]

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    width=0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=ones_count_normalized,
                    y=message_categories_list,
                    name="1 - category present",
                    orientation="h",
                    width=0.9
                ),
                Bar(
                    x=zeros_count_normalized,
                    y=message_categories_list,
                    name="0 - category absent",
                    orientation="h",
                    width=0.9
                )
            ],

            'layout': {
                'title': 'Data imbalance distribution, Total messages: {}'.format(labels_data.shape[0]),
                'yaxis': {
                    'title': "Message category"
                },
                'xaxis': {
                    'title': "Fraction"
                },
                'barmode': "stack",
                'automargin': "False",
                'height':650,
                'margin':dict(l=160, r=60, t=60, b=55)
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = sorted(list(zip(df.columns[4:], classification_labels)), key=lambda x: x[1], reverse=True)

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
