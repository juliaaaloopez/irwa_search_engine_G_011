import pandas as pd

from myapp.search.objects import Document
from typing import List, Dict


def load_corpus(path) -> List[Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    if path.endswith(".json"):
        df = pd.read_json(path)
    else:
        df = pd.read_csv(path)
        df = df.fillna("")
    return _build_corpus(df)

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus from dataframe
    :param df:
    :return:
    """
    corpus = {}
    for _, row in df.iterrows():
        doc = Document(**row.to_dict())
        corpus[doc.pid] = doc
    return corpus

