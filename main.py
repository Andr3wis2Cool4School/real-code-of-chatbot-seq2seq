import pandas as pd 
import numpy as np
from utils.dataloader import fit_text
from utils.summarizer import Seq2seqGloveSummarizer
from sklearn.model_selection import train_test_split
from utils.plot_tool import plot_and_save_history


def main():
    
    # make sure numpy is the same 
    np.random.seed(42)
    data_dir_path = './data'
    report_dir_path = './report'
    model_dir_path = './model'
    embedding_path = './embedding/glove.42B.300d.txt'

    print('We are Loading the csv file now....')
    data = pd.read_csv(data_dir_path + '/cons.csv')
    print('Finished loading data...')

    print('Extract X and y from the Dataframe now...')
    data = data[data['1'].notnull()]
    data = data[data['2'].notnull()]
    X = data['1']
    Y = data['2']

    config = fit_text(X, Y)

    print('Config extracted from input texts ...')
    summarizer = Seq2seqGloveSummarizer(config)
    summarizer.load_glove(embedding_path)

    print('Finished to load Emebedding from {} ....'.format(embedding_path))

    '''
    Using Sklearn package train_test_split 
    to split twice 
    get train, dev, test
    '''
    Xtrain, Xdt, Ytrain, Ydt = train_test_split(X, Y, test_size=0.4, random_state=42)
    Xdev, Xtest, Ydev, Ytest = train_test_split(Xdt, Ydt, test_size=0.5, random_state=42)

    print('Training Size: ', len(Xtrain))
    print('Validation Size: ', len(Xdev))

    print('Starting training....')
    history = summarizer.fit(Xtrain, Ytrain, Xdev, Ydev, epochs=1, batch_size=1)

    history_plot_file_path = report_dir_path + '/' + Seq2seqGloveSummarizer.model_name + '-history.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()