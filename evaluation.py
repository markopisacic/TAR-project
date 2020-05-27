from pathlib import Path
from flair.data import  Corpus
from flair.datasets import ColumnCorpus, DataLoader, ColumnDataset
from flair.models import SequenceTagger
import torch

torch.cuda.empty_cache()

MODEL_PATH = './resources/taggers/propaganda/best-model.pt'
TEST_PATH = './data/test.txt'
OUT_PATH = './resources/taggers/propaganda/eval.out'    # uncomment to save predictions for each sample to file

columns = {0: 'text', 1: 'propaganda', 2: 'doc_id', 3: 'sentence_id'}

model : SequenceTagger = SequenceTagger.load(MODEL_PATH)

test = ColumnDataset(Path(TEST_PATH), columns)
test = Corpus._filter_empty_sentences(test)

res, e = model.evaluate(DataLoader(test), OUT_PATH)

print("Main score: " + str(res.main_score) + ", loss: " + str(e))
print(res.log_header)
print(res.log_header)
print(res.detailed_results)
