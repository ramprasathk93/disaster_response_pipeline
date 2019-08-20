import json
import warnings
import plotly
import pandas as pd
import matplotlib
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine
from sklearn.externals import joblib
warnings.filterwarnings('ignore')


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    messages = df["message"]
    text = tokenize(' '.join(messages))
    wordcloud_json = plotly_wordcloud(text)
    barplot_json = stack_barplot(df)

    # create visuals
    graphs = [
        barplot_json,
        wordcloud_json
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
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def plotly_wordcloud(text):
    full_string = (" ").join(text)
    wc = WordCloud(stopwords=set(STOPWORDS),
                   max_words=200,
                   max_font_size=100)
    wc.generate(full_string)

    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []

    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    # get the positions
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])

    # get the relative occurence frequencies
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i * 100)

    wordcloud_json = {
        'data': [
            Scatter(
                x=x,
                y=y,
                textfont=dict(size=new_freq_list,
                              color=color_list),
                hoverinfo='text',
                hovertext=['{0}{1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                mode='text',
                text=word_list
            )
        ],

        'layout': {
            'title': 'Word Cloud of frequent words',
            'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
            'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}
        }
    }

    return wordcloud_json

def stack_barplot(df):
    categories = df.iloc[:, 4:].sum().sort_values(ascending=False)
    genres = df.groupby('genre').sum()[categories.index[:10]]
    data = []
    for cat in genres.columns[1:]:
        data.append(Bar(
            x=genres.index,
            y=genres[cat],
            name=cat)
        )

    barplot_json = {
        'data': data,
        'layout': {
            'title': 'Categories per genre (Top 10)',
            'xaxis': {'title': 'Genres', 'tickangle': 45},
            'yaxis': {'title': '# of Messages per Category', 'tickfont': {'color': 'color_bar'}},
            'barmode': 'stack'
        }
    }

    return barplot_json

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()