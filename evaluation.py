from flair.data import  Corpus
from flair.datasets import ColumnCorpus, DataLoader
from flair.models import SequenceTagger
import torch

torch.cuda.empty_cache()

MODEL_PATH = './resources/taggers/propaganda/best-model.pt'
OUT_PATH = './resources/taggers/propaganda/eval.out'    # uncomment to save predictions for each sample to file

model : SequenceTagger = SequenceTagger.load(MODEL_PATH)

columns = {0: 'text', 1: 'propaganda'}
data_folder = './data'

corpus : Corpus = ColumnCorpus(data_folder, columns, in_memory=False,
                               train_file='train.txt',
                               test_file='test.txt',
                               dev_file='validate.txt').downsample(0.2)

res, e = model.evaluate(DataLoader(corpus.test), OUT_PATH)

print("Main score: " + str(res.main_score) + ", loss: " + str(e))
print(res.log_header)
print(res.log_header)
print(res.detailed_results)
