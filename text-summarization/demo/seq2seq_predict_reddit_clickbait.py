from __future__ import print_function

import pandas as pd
from keras_text_summarization.library.seq2seq import Seq2SeqSummarizer
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = 'demo/data'
    model_dir_path = 'demo/models'

    print('loading csv file ...')
    # df = pd.read_csv(data_dir_path + "/fnon-clickbait.csv")
    df = pd.read_csv(data_dir_path + "/clickbait.csv", sep="|")

    X = df.text
    Y = df.title

    config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for i in range(len(X)):
        x = X[i]
        actual_headline = Y[i]
        headline = summarizer.summarize(x)
        # print('Article: ', x)
        print(i)        
        print('Original Headline: ', actual_headline)
        print('Generated Headline: ', headline)
        # print('Actual Text:',x)
        print("-------------------------------------")


if __name__ == '__main__':
    main()