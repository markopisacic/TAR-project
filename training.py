from typing import List

from flair.data import Corpus
from flair.datasets import ColumnCorpus

# define columns
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, FlairEmbeddings

columns = {0: 'text', 1: 'propaganda'}

# this is the folder in which train, test and dev files reside
data_folder = './data'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='validate.txt').downsample(0.4)
print(corpus)

tag_type = 'propaganda'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

embedding_types: List[TokenEmbeddings] = [

    WordEmbeddings('glove'),

    # CharacterEmbeddings(),

    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/propaganda',
              learning_rate=0.1,
              mini_batch_size=20,
              max_epochs=2,
              checkpoint=True)

# 8. plot weight traces (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_weights('resources/taggers/propaganda/weights.txt')


# to be plotted after training

# from flair.visual.training_curves import Plotter
# plotter = Plotter()
# plotter.plot_training_curves('loss.tsv')
# plotter.plot_weights('weights.txt')