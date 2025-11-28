import pandas as pd

from myapp.search.objects import Document
from typing import List, Dict
import ast


def load_corpus(path) -> List[Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    try:
        if path.endswith('.csv'):
            df = pd.read_csv(path)
        else:
            df = pd.read_json(path, orient='records')
    except ValueError:
        # Fallback for JSON files that are not in records format
        df = pd.read_json(path)

    corpus = _build_corpus(df)
    return corpus

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus from dataframe
    :param df:
    :return:
    """
    corpus = {}

    def parse_list_string(val):
        if isinstance(val, list): 
            return val
        if isinstance(val, str) and val.strip().startswith('['):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return []
        return []
    
    if 'processed_text' in df.columns:
        df['processed_text'] = df['processed_text'].apply(parse_list_string)
    
    if 'attributes' in df.columns:
        df['attributes'] = df['attributes'].apply(parse_list_string)

    df['tokens'] = df.apply(
        lambda row: (row['processed_text'] if isinstance(row['processed_text'], list) else []) + 
                    (row['attributes'] if isinstance(row['attributes'], list) else []), 
        axis=1
    )

    df = df.where(pd.notnull(df), None)
    
    for _, row in df.iterrows():
        try:
            # Convert row to Document
            doc_data = row.to_dict()
            
            # Ensure tokens are assigned
            doc = Document(**doc_data)
            
            # Store in corpus
            corpus[doc.pid] = doc
        except Exception as e:
            print(f"Error loading doc {row.get('pid', 'unknown')}: {e}")
            continue
        
    print(f"Loaded {len(corpus)} documents correctly with tokens.")
    return corpus

    

    

