import numpy as np
import pandas as pd
import re
import itertools
from collections import Counter
from sklearn import preprocessing
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import s3fs
import sys
import os
from nltk.corpus import stopwords
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'^https?:\/\/.*[\r\n]*', '', string) # addition #flags=re.MULTILINE
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def text_preprocessing(x_text, MAX_NB_WORDS, MAX_LEN_DOC, SPLIT_INDEX, INSPECTION_ROW=4):
    
    print ("Clean up texts...")
    
    # data type
    x_text = x_text.astype(str)
    
    if isinstance(x_text, pd.Series):
        x_text = x_text.values
    
    clean_func = np.vectorize(lambda x: clean_str(x))
    x_text = clean_func(x_text)
    
    print ("Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(x_text)
    
    word2id = tokenizer.word_index
    id2word = {v:k for k, v in word2id.items()}
    
    vocab_size = len(word2id.keys()) + 1

    print ("Integer encoding...")
    x_text = tokenizer.texts_to_sequences(x_text) # might need to change if not in pd.df
    
    print ("Padding...")
    x_text = sequence.pad_sequences(x_text, maxlen=MAX_LEN_DOC, dtype='int32',
                                   padding="pre", truncating="post")
    
    return x_text, vocab_size, id2word, word2id

def label_processing(y):
    
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    
    num_class = len(np.unique(y))
    y = np_utils.to_categorical(y, num_class)
    
    return [y, le]


def load_data(MAX_LEN_DOC=100, NUM_SUMPLE=5000, MAX_NB_WORDS=5000, 
              url1='s3://smart-newsdev-dmp/tmp/data/classification/data.csv',
              url2='s3://smart-newsdev-dmp/tmp/data/classification/data.csv',
             TEXT_TITLE_COMBINED=False): # if false use only 'title'
    """
    
    """
    # Load data from files
    print("Loading data...")
    train = pd.read_csv(url1, sep='|', error_bad_lines=False)
    test = pd.read_csv(url2, sep='|', error_bad_lines=False)
    split_index = train.shape[0]
    
    data = pd.concat([train, test])
        
    # limit samples
    # data = data.iloc[0:NUM_SUMPLE]
    
    # combine two columns
    if TEXT_TITLE_COMBINED:
        pre_x = data['title'] + " " + data['text']
    else:
        pre_x = data['title']
    
    # Generate encoded text sequence
    x_text, vocab_size, id2word, word2id = text_preprocessing(
        x_text=pre_x, MAX_NB_WORDS=MAX_NB_WORDS,
        MAX_LEN_DOC=MAX_LEN_DOC, SPLIT_INDEX=split_index)
    # print(ints2setences(x_text[5], id2word))
 
    # Generate labels
    y, labelEncoder= label_processing(data['category'])
    
    data
    
    return [x_text, y, split_index, vocab_size, id2word, word2id, pre_x, data, labelEncoder]

def ints2setences(sequence_array, id2word):
    result = []
    for int_enc in sequence_array:
        if int_enc != 0:
#             print(int_enc)
            result.append(id2word[int(int_enc)])
        
    return ' '.join(result)

def evaluation(y_pred, y_test, history):
    print ('---------Confusion Matrix Report -------------------- \n')
    print (sklearn.metrics.confusion_matrix(y_test, y_pred))
    print ('\n---------Classificaiton Report ---------------------- \n')
    print (sklearn.metrics.classification_report(y_test, y_pred))
    
    if history:
        print(history.history.keys())
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()