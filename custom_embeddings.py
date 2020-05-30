import pickle
from typing import Union, List
from flair.embeddings import TokenEmbeddings
from flair.data import Sentence
from transformers import AutoTokenizer
import torch

class NoContextBertEmbeddings(TokenEmbeddings):
    def __init__(self):
        self.name = 'NoContextBertEmbeddings'
        self.static_embeddings = True

        # TODO load with pickle
        self.embeddings = pickle.load(open('embeddings_2.pickle', 'rb'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.__embedding_length: int = 768
        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token in sentence:
                ids = self.tokenizer.encode(token.text, add_special_tokens = False)
                subtokens = self.tokenizer.convert_ids_to_tokens(ids)
                set_embedding = False
                for subtoken in subtokens:
                    if subtoken in self.embeddings:
                        token.set_embedding(self.name, torch.mean(self.embeddings[subtoken][-768*4:].reshape(4, 768), dim = 0))
                        set_embedding = True
                        break
                if not set_embedding:
                    token.set_embedding(self.name, torch.zeros(self.embedding_length, dtype=torch.float32))

        return sentences
