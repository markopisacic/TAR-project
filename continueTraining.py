from pathlib import Path
from flair.data import Corpus
from flair.datasets import WNUT_17
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
import sys
from flair.datasets import ColumnCorpus
from flair.trainers import ModelTrainer
import torch

if __name__ == '__main__':
    if(len(sys.argv) not in [2, 3]):
        raise ValueError('Usage: python resumeTraining.py <model_name> [checkpoint/best-model/..]')

    torch.cuda.empty_cache()

    model_name = sys.argv[1]
    which_model = 'checkpoint'
    if len(sys.argv) == 3:
        which_model = sys.argv[2]
    checkpoint = 'resources/taggers/' + model_name + '/' + which_model + '.pt'

    columns = {0: 'text', 1: 'propaganda', 2: 'doc_id', 3: 'sentence_id'}

    # this is the folder in which train, test and dev files reside
    data_folder = './data'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=False,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='validate.txt')

    trainer = ModelTrainer.load_checkpoint(checkpoint, corpus)
    trainer.train('resources/taggers/' + model_name,
                  learning_rate=0.05,
                  anneal_factor = 0.2,
                  train_with_dev = False,
                  mini_batch_size=10,
                  max_epochs=5,
                  checkpoint=True,
                  embeddings_storage_mode = 'gpu',
                  patience=0,
                  anneal_with_restarts = True,
                  monitor_test=True)

