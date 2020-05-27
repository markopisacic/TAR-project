import sys
from typing import List
import torch

from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        raise TypeError('Usage: python training.py <model_name>')
    model_name = sys.argv[1]
    
    torch.cuda.empty_cache()

    columns = {0: 'text', 1: 'propaganda', 2: 'doc_id', 3: 'sentence_id'}

    # this is the folder in which train, test and dev files reside
    data_folder = './data'

    # init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns, in_memory=False,
                                  train_file='train.txt',
                                  test_file='test.txt',
                                  dev_file='validate.txt')
    
    #corpus.filter_empty_sentences()
    #print(corpus)
    #for i in range(len(corpus.train)):
    #    if len(corpus.train[i]) == 1: #and len(corpus.train[i].get_token(1).text) == 0:
    #        print(corpus.train[i])
    #        print(len(corpus.train[i].get_token(1).text))
    #        sys.exit(0)

    tag_type = 'propaganda'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)

    embedding_types: List[TokenEmbeddings] = [

        #WordEmbeddings('glove'),

        # other embeddings

        # CharacterEmbeddings(),
        TransformerWordEmbeddings('bert-base-uncased'),
        #FlairEmbeddings('news-forward'),
        #FlairEmbeddings('news-backward'),
    ]
    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    # 5. initialize sequence tagger
    from flair.models import SequenceTagger

    tagger: SequenceTagger = SequenceTagger(hidden_size=50,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            train_initial_hidden_state = True,
                                            loss_weights = {'0': 1, '1': 20},
                                            use_crf=False,
                                            dropout = 0.2)

    # 6. initialize trainer
    from flair.trainers import ModelTrainer

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    # 7. start training
    trainer.train('resources/taggers/' + model_name,
                  learning_rate=0.05,
                  train_with_dev = False,
                  mini_batch_size=10,
                  max_epochs=1000,
                  checkpoint=True,
                  embeddings_storage_mode = 'gpu',
                  patience=2)

    # 8. plot weight traces (optional)
    from flair.visual.training_curves import Plotter

    #plotter = Plotter()
    #plotter.plot_weights('resources/taggers/propaganda/weights.txt')
    #plotter.plot_training_curves('resources/taggers/propaganda/loss.tsv')
