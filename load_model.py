import os

from sentence_transformers import SentenceTransformer


def get_embedder():
    model_path = os.path.expanduser('~/projects/silvia/silvia-ai/src/submodules/ko_sentencebert/output/training_sts')
    return SentenceTransformer(model_path, device='cpu')
