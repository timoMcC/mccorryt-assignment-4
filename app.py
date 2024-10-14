from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# dataset
newsgroups = fetch_20newsgroups(subset='all')
documents = newsgroups.data

#vectorizer and lsa using scikit svd and tf-idf
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
svd = TruncatedSVD(n_components=100) 
lsa_matrix = svd.fit_transform(tfidf_matrix)

def search_engine(query):
    query_vector = vectorizer.transform([query])
    query_lsa = svd.transform(query_vector)

    similarities = cosine_similarity(query_lsa, lsa_matrix).flatten()
    indices = similarities.argsort()[-5:][::-1]  
    top_documents = [documents[i] for i in indices]
    top_similarities = similarities[indices].tolist()

    return top_documents, top_similarities, indices.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
