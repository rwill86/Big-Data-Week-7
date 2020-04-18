#!/usr/bin/python

# Try to open imports
try:
    import sys
    import random
    import math
   import os
    import time
    import numpy as np
	import pandas as pd
    from matplotlib import pyplot as plt
	from sklearn.decompostion import PCA as sklearnPCA
	from numpy import dot
	from numpy.linalg import norm
	import json
	from nltk.tokenize import word_tokenize
	import re
	from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrices.pairwise import linear_kernel
	from sklearn.feature_extraction.text import CountVectorizer
	# Sentiment Analysis
    from nltk.stem.snowball import FrenchStemmer
	from nltk.classify import NaiveBayesClassifier
    from nltk.corpus import subjectivity
    from nltk.sentiment import SentimentAnalyzer
    from nltk.sentiment.util import *
	# Vader
	from nltk.sentiment.vader import SentimentIntensityAnalyzer
	from nltk import tokenize

# Error when importing
except ImportError:
    print('### ', ImportError, ' ###')
    # Exit program
    exit()

#tokenize
def tokenize(s):
    return tokens_re.findall(s)	
	
#Cosine Sim
def cosine_sim(u, v):
    return dot(u, v) / (norm(v) * norm(u))

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

#Average word vectors
def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
	nwords = 0
	for word in words:
	   if word in vocabulary:
	       nwords = nwords + 1
		   feature_vector = np.add(feature_vector, model[word])
    if nwords:
	    feature_vector = np.divide(feature_vector, nwords)
    retrun feature_vector
	
	
def average_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
	features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features) for tokenized_sentence in corpus]
    retrun np.array(features)
	
	
# Read input
def read():
    # Word embedding
	vocabulary = ['King', 'Man','Queen','Woman']
	tokens = {w:i for i, w in enumerate(vocabulary)}
	N = len(vocabulary)
	w = np.zeros((N, N))
	np.fill_diagonal(w, 1)
	print("Cosine Similarity ".format(cosine_sim(w['King'], w['woman'])))
	
	Q = "tesla nasa"
	tfidf = TfidfVectorizer(analyer='word', ngram_range=(1,1), min_df = 1, stop_words = 'english', max_features=500)
    features = tfidf.fit(original_documents)
	cporus_tf_idf = tfidf.transform(original_documents)
	sum_words = corpus_tf_idf.sim(axis = 0)
	words_freq = [(word, sum_words[0, indx]) for word, idx in tfidf.vocabulary_.items()]
	print(sorted(wrods_freq, key = lambda x: x[1], reverse = true)[:5])
	print('test', corpus_tf_idf[1], features.vocabulary_['tesla'])
	
	new_features = tfidf.transform([query
	cosine_similarities = linear_kernel(new_features, corpus_tf_idf).flatten()
	related_docs_indices = cosine_similarities.argsort()[::-1]
	topk = 5
	print('Top - {0} documents'.format(topk))
	for i in range(topk):
	   print(i, original_documents[related_docs_indices[i]])
	   
    stemmer = FrenchStemmer()
    analyzer = CountVectorizer().build_analyzer()
	stem_vectorizer = CountVectorizer(analyzer=stemmed_words)
    print(stem_vectorizer.fit_transform(['Tu marches dans la rue']))
    print(stem_vectorizer.get_feature_names())
	
	tweet = 'RT @marcobonzanini: just an example! :D http://example.com #NLP'
    print(word_tokenize(tweet))

	
	with open('mytweets.json', 'r') as f:
    line = f.readline() 
    tweet = json.loads(line) 
    print(json.dumps(tweet, indent=4))
	
	with open('mytweets.json', 'r') as f:
    for line in f:
        tweet = json.loads(line)
        tokens = preprocess(tweet['text'])
        do_something_else(tokens)	   
	# Sentiment Analysis
	n_instances = 100
	subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
	obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
	sd = len(subj_docs)
	od = len(obj_docs)
	subj_docs[0]
	train_subj_docs = subj_docs[:80]
	test_subj_docs = subj_docs[80:100]
	train_obj_docs = obj_docs[:80]
	test_obj_docs = obj_docs[80:100]
	training_docs = train_subj_docs + train_obj_docs
    testing_docs = test_subj_docs + test_obj_docs
	sentim_analyzer = SentimentAnalyzer()
    all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])
	unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
	uf = len(unigram_feats)
	# Classification
	sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
	training_set = sentim_analyzer.apply_features(training_docs)
	test_set = sentim_analyzer.apply_features(testing_docs)
	trainer = NaiveBayesClassifier.train
	classifier = sentim_analyzer.train(trainer, training_set)
	for key,value in sorted(sentim_analyzer.evaluate(test_set).items()):
	    print('{0}: {1}'.format(key, value))	
    # Vader	
	words = ['ice', 'police', 'global']
	document = 'ice is melting due to global warming'.split()
	sorted(extract_unigram_feats(document, words).items())
	
	sentences = ["VADER is smart, handsome, and funny.", "VADER is smart, handsome, and funny!"]
	lines_list = tokenize.sent_tokenize(paragraph)
	sentences.extend(lines_list)
	
	tricky_sentences = ["The movie was too good", "This movie was actually neither that funny, nor super witty."]
	sentences.extend(tricky_sentences)
	sid = SentimentIntensityAnalyzer()
	for sentence in sentences:
	    ss = sid.polarity_scores(sentence)
		for k in sorted(ss):
		   print('{0}: {1}, '.format(k, ss[k]), end='')
		print()
		

# Main
def main():
    # Read Input
    read()
    # Close Program
    exit()


# init
if __name__ == '__main__':
    # Begin
    main()


