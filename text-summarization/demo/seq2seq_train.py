from __future__ import print_function

import pandas as pd
from sklearn.model_selection import train_test_split
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
from keras_text_summarization.library.applications.fake_news_loader import fit_text
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = 'demo/data'
    report_dir_path = 'demo/reports'
    model_dir_path = 'demo/models'

    print('loading csv file ...')
    df = pd.read_csv(data_dir_path + "/fake_or_real_news.csv")

    print('extract configuration from input texts ...')
    Y = df.title
    X = df['text']

    config = fit_text(X, Y)

    summarizer = Seq2SeqSummarizer(config)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

    print('demo size: ', len(Xtrain))
    print('testing size: ', len(Xtest))

    print('start fitting ...')
    history = summarizer.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=100, model_dir_path=model_dir_path)


if __name__ == '__main__':
    main()
