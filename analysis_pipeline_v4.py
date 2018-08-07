#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Basic Analysis Pipeline
# Kelly Geyer, Hao Wang, Arjoon Srikanth


# display figure in jupyter notebook
# get_ipython().magic(u'matplotlib inline')





# Numerical packages
import numpy as np
from scipy.sparse import coo_matrix, hstack, csr_matrix

# NLP packages
import nltk
import ll_author_id as lai

# ML packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import sklearn.metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from nltk.stem import WordNetLemmatizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
wordnet_lemmatizer = WordNetLemmatizer()

# Plotting
import matplotlib.pyplot as plt

# General packages
import os
import sys
import json
import re
import random
import itertools
from collections import defaultdict





# ## General Functions to make analysis easier
def update_doc_json(dict_obj, doc_id):
    '''
    This function updates the document json with term frequencies
    '''
    fn = os.path.join(data_dir, 'article_' + str(doc_id) + '.json')
    with open(fn, "w") as json_file:
        json.dump(dict_obj, json_file)



# ## Normalize Text
# In this section we 1. convert UTF8 characters to ASCII (assuming that all docs are English) and 2. remove punctuation
# Functions used for normalizing text
def normalize_4_tfidf(doc_ids):
    '''
    This function normalizes the articles and titles, then updates the json

    Note that stop words are removed in TfidfVectorizer

    :param doc_ids: list of document IDs
    :return data: data containing new fields
    '''
    cachedStopWords = nltk.corpus.stopwords.words("english")
    for ii in doc_ids:
        # Normalize text using LiLAC (only when it hasn't already been done!) and remove stop words
        if 'norm_article_text' not in data[ii].keys():
            art_text = normalize_rtgVersion(data[ii]['article_text'])
            data[ii]['norm_article_text'] = art_text
        if 'norm_article_title' not in data[ii].keys():
            title_text = normalize_rtgVersion(data[ii]['article_title'])
            data[ii]['norm_article_title'] = title_text
        if 'norm_article_char_text' not in data[ii].keys():
            art_char_text = normalize_char_rtgVersion(data[ii]['article_text'])
            data[ii]['norm_char_article_text'] = art_char_text
        if 'norm_article_char_title' not in data[ii].keys():
            title_char_text = normalize_char_rtgVersion(data[ii]['article_title'])
            data[ii]['norm_char_article_title'] = title_char_text
        # Tokenize
        # article_tokens = nltk.word_tokenize(norm_article)
        # title_tokens = nltk.word_tokenize(norm_title)b
        # Save to doc .json file
        update_doc_json(data[ii], ii)

def normalize_char_rtgVersion(ln):
    """
    This function normalizes text for character ngram analysis
    """
    # padding punctuation with \n
    # ln = re.sub(r'([,.!?])([@\#A-Za-z0-9])', r'\1\n\2',ln)  # Insert '\n'
    ln = re.sub(r'([,.!?])([@\#A-Za-z0-9])', r'\1 ', ln)
    return ln

def normalize_rtgVersion(ln):
    """
    This function normalizes text, normalization options include
    1. convert UTF8 to ASCII
    2. Remove nonsentenial punctuation
    3. Remove mark up
    4. remove word punctuation
    5. remove repeats (i.e. hahahahaha)
    6. remove twitter media, handles, tags

    :param ln: Text before normalization process
    :return ln: Normalized text
    """
    # Load dictionary for ASCII conversion
    rw_hash = lai.create_utf8_rewrite_hash()
    # Various normalization routines -- pick and choose as needed
    ln = lai.convertUTF8_to_ascii(ln, rw_hash)
    ln = remove_nonsentential_punctuation_rtgVersion(ln)
    ln = lai.remove_markup(ln)
    ln = lai.remove_word_punctuation(ln)
    ln = lai.remove_repeats(ln)           # Remove repeated laughter (i.e. hahahahaha)
    ln = ln.lower()                       # Convert to lower case
    ln = lai.remove_twitter_meta(ln)
    ln = re.sub('\s+', ' ', ln)
    if (ln == ' '):
        ln = ''
    return ln

def remove_nonsentential_punctuation_rtgVersion(ln):
    """
    This function removes nonsentenial punctuation from text.
    :param/return ln: String

    TODO:
    0. Check to see that TfidfVectorizor does not already do 1-5.
    If it doesn't do 1-5 as necessary
    1. Remove '&'
    2. Remove '.'
    3. Remove '''
    4. Remove ',' (special consideration for #s vs in sentences)
    5. Remove all '/'
    """
    # remove '-'
    ln = re.sub('^\-+', '', ln)
    ln = re.sub('\-\-+', '', ln)
    ln = re.sub('\s\-+', '', ln)
    # remove '~'
    ln = re.sub('\~', ' ', ln)
    # remove standard double quotes
    ln = re.sub('\"', '', ln)
    # remove single quotes
    ln = re.sub("^\'+", '', ln)
    ln = re.sub("\'+$", '', ln)
    ln = re.sub("\'+\s+", ' ', ln)
    ln = re.sub("\s+\'+", ' ', ln)
    ln = re.sub("\s+\`+", ' ', ln)
    ln = re.sub("^\`+", ' ', ln)
    # remove ':'
    ln = re.sub("\:\s", " ", ln)
    ln = re.sub("\:$", "", ln)
    # remove ';'
    ln = re.sub('\;\s', ' ', ln)
    ln = re.sub('\;$', '', ln)
    # remove '_'
    ln = re.sub('\_+\s', ' ', ln)
    ln = re.sub('^\_+', '', ln)
    ln = re.sub('_+$', '', ln)
    ln = re.sub('\_\_+', ' ', ln)
    # remove ','
    ln = re.sub('\,+([\#A-Za-z])', ' \g<1>', ln)
    ln = re.sub('\,+$', ' ', ln)
    ln = re.sub('\,\.\s', ' ', ln)
    ln = re.sub('\,\s', ' ', ln)
    # remove '*'
    ln = re.sub('\s\*+', ' ', ln)
    ln = re.sub('\*+\s', ' ', ln)
    ln = re.sub('\*\.', ' ', ln)
    ln = re.sub('\s\*+\s', ' ', ln)
    ln = re.sub('^\*+', '', ln)
    ln = re.sub('\*+$', '', ln)
    # Keep only one '.', '?', or '!'
    ln = re.sub('\?[\!\?]+', '?', ln)
    ln = re.sub('\![\?\!]+', '!', ln)
    ln = re.sub('\.\.+', '.', ln)
    # # remove '/'
    ln = re.sub('\s\/', ' ', ln)
    ln = re.sub('\/\s', ' ', ln)
    # remove sentence final '!' and '?'
    ln = re.sub('[\!\?]+\s*$', '', ln)
    # remove other special characters
    ln = re.sub('\|', ' ', ln)
    ln = re.sub(r'\\', ' ', ln)
    # Remove parentheses that are not part of emoticons.
    # Note sure of the best way to do this, but here's a conservative
    # approach.
    ln = re.sub('\(([@\#A-Za-z0-9])', '\g<1>', ln)
    ln = re.sub('([@\#A-Za-z0-9])\)', '\g<1> ', ln)
    # Clean up extra spaces
    ln = re.sub('^\s+', '', ln)
    ln = re.sub('\s+$', '', ln)
    ln = re.sub('\s+', ' ', ln)


    return ln




def get_stemmers_lemmatizers_rtgVersion(word_list):
    """
    This function takes a list of words and only returns stemmers and lemmatizers in the list

    :param word_list list of words
    :return filtered_list list of stemmers and lemmatizers from the original word list
    """
    return [wordnet_lemmatizer.lemmatize(i) for i in word_list]


# ## Create Design matrix

# Function for computing TF-IDF

def create_vocab_dict(train_ids, test_ids, key_name, analyzer_type, ngram_len):
    '''
    This function creates a vocabulary list

    :param train_ids: List of training IDs
    :param test_ids: List of testing IDs
    :param key_name: Key of dictionary to perform TF-IDF
    :param analyzer_type: type of ananlyzer to be used word/char
    :param n_gram_len: ngram length
    :return tfidf_obj: TfidfVectorize object for entire corpus
    '''
    # Get vocabulary list from training set
    if analyzer_type == 'word':
        trObj = TfidfVectorizer(strip_accents='unicode', analyzer=analyzer_type, stop_words='english',
                                ngram_range=(ngram_len, ngram_len))
        trObj.fit_transform(map(lambda x: data[x][key_name], train_ids))
        trVocab = trObj.get_feature_names()

        # Get vocabulary list from testing set
        tsObj = TfidfVectorizer(strip_accents='unicode', analyzer=analyzer_type, stop_words='english',
                                ngram_range=(ngram_len, ngram_len))
        tsObj.fit_transform(map(lambda x: data[x][key_name], test_ids))
        tsVocab = tsObj.get_feature_names()

        # for word as analyzer, only keep the stemmer of the list of vocabs
        trVocab = get_stemmers_lemmatizers_rtgVersion(trVocab)
        tsVocab = get_stemmers_lemmatizers_rtgVersion(tsVocab)
    else:
        trObj = TfidfVectorizer(strip_accents='unicode', analyzer=analyzer_type,
                                ngram_range=(ngram_len, ngram_len))
        trObj.fit_transform(map(lambda x: data[x][key_name], train_ids))
        trVocab = trObj.get_feature_names()

        # Get vocabulary list from testing set
        tsObj = TfidfVectorizer(strip_accents='unicode', analyzer=analyzer_type,
                                ngram_range=(ngram_len, ngram_len))
        tsObj.fit_transform(map(lambda x: data[x][key_name], test_ids))
        tsVocab = tsObj.get_feature_names()



    # Create vocabulary dictionary where keys are terms and values are indexes
    all_vocab = list(set(trVocab).intersection(set(tsVocab)))
    vocab_dict = {all_vocab[ii]: ii for ii in range(len(all_vocab))}

    # Create TfidfVectorize Object
    tfidf_obj = TfidfVectorizer(strip_accents='unicode', analyzer='word', stop_words='english', vocabulary=vocab_dict)
    return tfidf_obj

def compute_tfidf(train_ids, test_ids, key_name,analyzer_type,ngram_len):
    '''
    This function computes TF-IDF for a list of document IDs

    :param train_ids: list of train document ids
    :param test_ids: list of test doc ids
    :param key_name: Key of dictionary to perform TF-IDF
    :param n_gram_len: ngram length
    :return Xtr: training design matrix
    :return Xts: testing design matrix

    '''
    # Create doc2idx dictionary where keys are doc IDs and values are IDX
    train_doc2idx = {train_ids[i] : i for i in range(len(train_ids))}
    test_doc2idx = {test_ids[i] : i for i in range(len(test_ids))}
    # Create TfidfVectorize object
    tfidf_obj = create_vocab_dict(train_ids, test_ids, key_name,analyzer_type,ngram_len)
    # Create training design matrix
    Xtr = tfidf_obj.fit_transform(map(lambda x: data[x][key_name], train_ids))
    # Create testing design matrix
    Xts = tfidf_obj.fit_transform(map(lambda x: data[x][key_name], test_ids))
    return (Xtr, train_doc2idx, Xts, test_doc2idx)





# ## Create Response Matrix

# Functions for creating response matrices
def create_response_matrix(doc_ids):
    '''
    This function returns a multilabel sparse CSR matrix

    :param doc_ids: documnet ids to be processed
    :return Y: the response matrix
    '''
    ll = map(lambda x: set(labels[unicode(x)]), doc_ids)
    mlb = preprocessing.MultiLabelBinarizer(sparse_output=True, classes=all_labels)
    Y = mlb.fit_transform(ll)
    return Y








# ## Fit Classification Model


def fit_classification_model(model,Xtr,Ytr,Xts,Yts):
    """

    :param model: the model to be used to fit values, possible options:[naive bayesian, logistic, svm, lasso, ridge]
    :param Xtr: training design matrix
    :return Xts: testing design matrix
    :param Ytr: training response matrix
    :param Yts: testing response matrix
    :return: a matrix of y_pred
    """



    # Multinomial Naive Bayesian

    if model == 'naive_bayesian':
        gnb = OneVsRestClassifier(GaussianNB())
        fit = gnb.fit(Xtr, Ytr)
        y_pred= cross_val_predict(fit,Xts,Yts)



    # Logistic Regression

    if model == 'logistic' :
        logreg = linear_model.LogisticRegression(C=1e5, multi_class='ovr')
        logreg = OneVsRestClassifier(linear_model.LogisticRegression(C=1e5, multi_class='ovr'))
        fit = logreg.fit(Xtr, Ytr)
        y_pred = cross_val_predict(fit,Xts,Yts)


    # one vs all SVM
    if model =='svm':
        # Convert to CSR sparse matrix...best for SVM
        Xtr_sparse = csr_matrix(Xtr)
        Xts_sparse = csr_matrix(Xts)
        # Learn to predict each class against the other
        classifier = OneVsRestClassifier(svm.SVC(probability=True))
        fit = classifier.fit(Xtr_sparse, Ytr)
        y_pred = fit.predict(Xts)
    # lasso
    if model == 'lasso':
        lasso = linear_model.Lasso()
        y_pred = cross_val_predict(lasso, Xts, Yts)
    # ridge
    if model == 'ridge':
        ridge = linear_model.Ridge()
        y_pred = cross_val_predict(ridge, Xts, Yts)
    else:
        print "Model not implemented, please choose from the following: 'naive_bayesian', 'logistic', 'svm', 'lasso', 'ridge'"
    return y_pred


# ## Evaluation
def make_evaluation_plots(Yts, y_score, all_labels):
    """

    :param Yts: Y test matrix
    :param y_score: prediction y score
    :param all_labels: all labels to be classified
    :return: plots of evaluation results
    """
    # Compute ROC curve and ROC area for each class
    fpr = dict({})
    tpr = dict({})
    roc_auc = dict({})
    for i in range(len(all_labels)):
        fpr[i], tpr[i], _ = roc_curve(Yts[:, i], y_score[:, i], pos_label=1)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create colormap for plot
    nplots = len(all_labels)
    colormap = plt.cm.hsv
    colors = [colormap(i) for i in np.linspace(0, 1, nplots)]
    # colors = [colormap(i) for i in np.linspace(0, 0.9, nplots)]
    lw = 2
    n_classes = nplots

    # Create ROC curves
    # plt.figure()
    plt.figure(figsize=(15, 8))
    for i, color in zip(range(nplots), colors):
        lab = 'ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i])
        print lab
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=lab)

    plt.plot([0, 10], [0, 10], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="best")

    plt.show()

    # save figure
    plt.savefig('ROC Curve')

    # Compute Precision-Recall and plot curve

    plt.figure(figsize=(15, 8))
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(all_labels)):
        precision[i], recall[i], _ = precision_recall_curve(Yts[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Yts[:, i], y_score[:, i])

    a = Yts.ravel()

    a = a.tolist()[0]
    b = y_score.ravel()
    b = b.tolist()

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = precision_recall_curve(a,
                                                                    b)
    average_precision["micro"] = average_precision_score(Yts, y_score,
                                                         average="micro")

    # Plot Precision-Recall curve
    plt.figure(figsize=(15, 8))

    plt.plot(recall[0], precision[0], lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()

    # save figure
    plt.savefig('Precision-Recall Curve')

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(15, 8))
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(len(all_labels)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    # save figure
    plt.savefig('Precision-Recall Curve')





### Recording nad saving the results of each experiment run: each evaluation e corresponding to the model used and the sample size



def record_evaluation_result(result_dict,model,y_true,y_pred):
    """
    This function modifies the input result_dict dictionary that records the result of each experiment
    :param result_dict: input dict where keys are teh models and sample size
    :param model: a string that specifies model used to fit
    :param y_true:
    :param y_pred:
    :return: result dict: the modified result dict with auc score, f1 score, recall, and precision score corresponding to each
    experiment run
    """

    # unravel input matrices to array
    a = y_true.ravel()
    a = a.tolist()[0]
    b = y_pred.ravel()
    b = b.tolist()

    # calculate various scores
    auc = sklearn.metrics.roc_auc_score(a,b)
    precision = sklearn.metrics.precision_score(a,b,average='weighted')
    recall = sklearn.metrics.recall_score(a, b, average='weighted')
    f1 = sklearn.metrics.f1_score(y_true, y_pred, average='weighted')

    result_dict[model]['auc'].append(auc)
    result_dict[model]['f1'].append(f1)
    result_dict[model]['precision'].append(precision)
    result_dict[model]['recall'].append(recall)

    return result_dict





data_dir = '/Users/ar_magnavox/Desktop/sample_data'  # Contains all data files
data_key_fn = 'article_key.json'

    # Load article key (links article IDs to filenames)
with open(os.path.join(data_dir, data_key_fn), 'r') as f:
    labels = json.load(f)

    # Load data and save into data dictionary (keys are article ids, values are dictionaries of features)
data = dict({})
fns = filter(lambda x: x != data_key_fn, os.listdir(data_dir))
for ii in fns:
    with open(os.path.join(data_dir, ii), 'r') as f:
        art = json.load(f)
    article_id = int(re.findall(r'\d+', ii)[0])
    data[article_id] = art
    # ## Split Data into testing and training sets
    # We split the data before forming the design matrix since some features are dependent on all documents

    # Randomly shuffle document IDs
all_ids = data.keys()
random.shuffle(all_ids)

# Test on a subset of data
all_ids = all_ids[0:100]
# Assign 30% to test set, rest to training set
nTest = int(.3 * len(all_ids))
test_ids = all_ids[0:nTest]
train_ids = all_ids[nTest:len(all_ids)]





all_labels = list([])
all_labels.extend(list(itertools.chain.from_iterable(map(lambda x: labels[unicode(x)], test_ids))))
all_labels.extend(list(itertools.chain.from_iterable(map(lambda x: labels[unicode(x)], train_ids))))
all_labels = list(set(all_labels))
print "all labels: ", all_labels

label2idx = {all_labels[i]: i for i in range(len(all_labels))}

def main():
    # Parameters
    # Define all parameters
    # data_dir = '/Users/ar_magnavox/Desktop/sample_data'  # Contains all data files
    # data_key_fn = 'article_key.json'
    #
    # # Load article key (links article IDs to filenames)
    # with open(os.path.join(data_dir, data_key_fn), 'r') as f:
    #     labels = json.load(f)
    #
    # # Load data and save into data dictionary (keys are article ids, values are dictionaries of features)
    # data = dict({})
    # fns = filter(lambda x: x != data_key_fn, os.listdir(data_dir))
    # for ii in fns:
    #     with open(os.path.join(data_dir, ii), 'r') as f:
    #         art = json.load(f)
    #     article_id = int(re.findall(r'\d+', ii)[0])
    #     data[article_id] = art
    # # ## Split Data into testing and training sets
    # # We split the data before forming the design matrix since some features are dependent on all documents
    #
    # # Randomly shuffle document IDs
    # all_ids = data.keys()
    # random.shuffle(all_ids)
    #
    # # Test on a subset of data
    # all_ids = all_ids[0:100]

    # Normalize documents for TF-IDF. This can take a while to run even if you've already created normalized text
    normalize_4_tfidf(all_ids)
    # TODO: Try on larger set (~1000s)


    # # Assign 30% to test set, rest to training set
    # nTest = int(.3 * len(all_ids))
    # test_ids = all_ids[0:nTest]
    # train_ids = all_ids[nTest:len(all_ids)]



    # Compute TF-IDF features and store data in a CSR sparse matrices
    # standard word tfidf
    (Xtr1, train_doc2idx1, Xts1, test_doc2idx1) = compute_tfidf(train_ids, test_ids, 'norm_article_title', 'word', 1)
    (Xtr2, train_doc2idx2, Xts2, test_doc2idx2) = compute_tfidf(train_ids, test_ids, 'norm_article_text', 'word', 1)
    # character ngram tfidf
    (Xtr3, train_doc2idx3, Xts3, test_doc2idx3) = compute_tfidf(train_ids, test_ids, 'norm_char_article_title', 'char',
                                                                3)
    (Xtr4, train_doc2idx4, Xts4, test_doc2idx4) = compute_tfidf(train_ids, test_ids, 'norm_char_article_text', 'char',
                                                                3)

    # TODO_done: Other features to try (built into TfidfVectorizer) are 1. character n-grams and
    # 2. word n-grams. character n-grams will require a different normalization method because we
    # will want punctuaion. Recall that we removed punctutation from word TF-IDF.
    #
    # TODO_done: Any feature applied to article_text can also be applied to article_title

    # Check indexes - values should be 0
    print cmp(train_doc2idx1, train_doc2idx2)
    print cmp(test_doc2idx1, test_doc2idx2)
    print cmp(train_doc2idx3, train_doc2idx4)
    print cmp(test_doc2idx3, test_doc2idx4)

    # Concatenate matrices
    Xtr = hstack([Xtr1, Xtr2, Xtr3, Xtr4])
    Xts = hstack([Xts1, Xts2, Xts3, Xts4])

    # Note that the # of rows should match
    # print Xts1.shape,Xts2.shape,Xts3.shape,Xts4.shape
    # print Xtr1.shape, Xtr2.shape, Xtr3.shape, Xtr4.shape

    # Remove unnescesary variables
    del Xtr1, Xtr2, Xts1, Xts2, train_doc2idx1, train_doc2idx2, test_doc2idx1, test_doc2idx2
    del Xtr3, Xtr4, Xts3, Xts4, train_doc2idx3, train_doc2idx4, test_doc2idx3, test_doc2idx4






    # Get all unique labels
    # all_labels = list([])
    # all_labels.extend(list(itertools.chain.from_iterable(map(lambda x: labels[unicode(x)], test_ids))))
    # all_labels.extend(list(itertools.chain.from_iterable(map(lambda x: labels[unicode(x)], train_ids))))
    # all_labels = list(set(all_labels))
    # print "all labels: ", all_labels

    # Create label2idx, where keys are labels and values are Y column index

    # label2idx = {all_labels[i]: i for i in range(len(all_labels))}

    # Create CSR sparse response matrices
    Yts = create_response_matrix(test_ids)
    Ytr = create_response_matrix(train_ids)
    # Number of columns should match
    print Yts.shape, Ytr.shape

    # ## Standardize design matrices

    # Standardize the data by ensuring that each column has mean=0 and standard deviation=1. Unfortunately this only
    # works on dense matrices
    # TODO: Also try data without standardization
    Xtr_std1 = preprocessing.StandardScaler().fit_transform(Xtr.todense())
    Xts_std1 = preprocessing.StandardScaler().fit_transform(Xts.todense())
    Xtr_std2 = preprocessing.MinMaxScaler().fit_transform(Xtr.todense())
    Xts_std2 = preprocessing.MinMaxScaler().fit_transform(Xts.todense())
    if not isinstance(Yts, np.matrixlib.defmatrix.matrix):
         Yts = Yts.todense()
    # make predictions based on model
    y_score = fit_classification_model('lasso',Xtr_std1,Ytr,Xts_std1,Yts)
    print y_score





    # make plots
    make_evaluation_plots(Yts,y_score,all_labels)

    # dictionary that will record evaluation results
    RESULT_DICT = defaultdict(lambda: defaultdict(list))

    RESULT_DICT = record_evaluation_result(RESULT_DICT,'svm',Yts,y_score)
    # save into a json file
    with open('evaluation_results', 'w') as outfile:
        json.dump(RESULT_DICT, outfile)

