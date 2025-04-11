from fastembed import TextEmbedding as Embedding
import torch
import numpy as np


class EmbeddingModel:
    def __init__(self):
        """
        - embed the model as BAAI/bge-small-en-v1.5
        - the model name can be optionally moved to a config
        """
        self.model = Embedding("BAAI/bge-small-en-v1.5")

    def embed(self, input_text, conver_to_tensor=False):
        """
        - generate embeddings for single input text
        """
        if conver_to_tensor:
            return torch.tensor(list(self.model.embed(input_text))[0])
        return list(self.model.embed(input_text))[0]

    def embedList(self, text_list, conver_to_tensor=False):
        """
        - generate embeddings for a list of input texts
        """
        if conver_to_tensor:
            np_array = np.array(list(self.model.embed(text_list)))
            return torch.tensor(np_array)
        return list(self.model.embed(text_list))