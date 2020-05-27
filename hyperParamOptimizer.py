from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.hyperparameter.param_selection import SequenceTaggerParamSelector, OptimizationValue
from flair.training_utils import EvaluationMetric
from hyperopt import hp

if __name__ == '__main__':
    columns = {0: 'text', 1: 'propaganda', 2: 'doc_id', 3: 'sentence_id'}

    # this is the folder in which train, test and dev files reside
    data_folder = './data'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=False,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='validate.txt')

    # define your search space
    search_space = SearchSpace()
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[
        WordEmbeddings('glove')
        # ,[FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]
    ])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[16, 32, 50, 64, 80, 100, 128])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.01, 0.025])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[5, 10, 20, 30])
    search_space.add(Parameter.USE_CRF, hp.choice, options=[False])

    param_selector = SequenceTaggerParamSelector(
        corpus,
        'propaganda',
        './resources/opt_results2/',
        5,
        EvaluationMetric.MEAN_SQUARED_ERROR,
        1,
        OptimizationValue.DEV_LOSS
    )

    param_selector.optimize(search_space, max_evals=20)
