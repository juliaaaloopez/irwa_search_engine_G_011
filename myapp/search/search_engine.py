import random
import numpy as np
import os
import json
from gensim.models import Word2Vec, Doc2Vec

from myapp.search.objects import Document

from myapp.search.algorithms import(
    search_tfidf,
    search_bm25,
    search_word2vec, 
    search_doc2vec,
    search_custom
)


def dummy_search(corpus: dict, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        new_doc = doc.model_copy(update={'score': round(random.random(), 4)})
        res.append(new_doc)
    return res


class SearchEngine:
    """Class that implements the search engine logic"""

    def __init__(self):
        # Definim rutes (relatives a on s'executa web_app.py)
        base_path = os.path.dirname(os.path.abspath(__file__)) # carpeta myapp/search/
        project_root = os.path.abspath(os.path.join(base_path, '..', '..')) # arrel del projecte
        
        # Rutes als fitxers generats en Parts 2 i 3
        # Ajusta aquestes rutes si els fitxers estan en un altre lloc
        self.paths = {
            "index": os.path.join(project_root, "project_progress", "part_2", "inverted_index.json"),
            "idf": os.path.join(project_root, "project_progress", "part_2", "idf_scores.json"),
            "w2v": os.path.join(project_root, "project_progress", "part_3", "my_word2vec_model.model"),
            "d2v": os.path.join(project_root, "project_progress", "part_3", "my_doc2vec_model.model")
        }

        print("Initializing Search Engine...")
        
        # 1. Carreguem Inverted Index
        self.inverted_index = self._load_json(self.paths["index"])
        
        # 2. Carreguem IDF Scores (TF-IDF)
        self.idf_scores = self._load_json(self.paths["idf"])
        
        # 3. Models Embeddings (Lazy loading o càrrega inicial)
        self.w2v_model = self._load_w2v_model()
        self.d2v_model = self._load_d2v_model()
        
        # 4. Pre-càlculs que depenen del corpus (s'inicialitzen buits)
        self.bm25_idf = {} 
        self.brand_ratings = {}
        self._corpus_processed = False

    
    def _load_json(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: File not found: {path}")
            return {}
        
        
    def _load_w2v_model(self):
        try:
            print(f"Loading Word2Vec model from {self.paths['w2v']}...")
            return Word2Vec.load(self.paths['w2v'])
        except Exception as e:
            print(f"Warning: Could not load Word2Vec model: {e}")
            return None
        

    def _load_d2v_model(self):
        try:
            print(f"Loading Doc2Vec model from {self.paths['d2v']}...")
            return Doc2Vec.load(self.paths['d2v'])
        except Exception as e:
            print(f"Warning: Could not load Doc2Vec model: {e}")
            return None
        

    def _prepare_corpus_stats(self, corpus):
        """Calcula estadístiques globals del corpus un sol cop."""
        if self._corpus_processed:
            return

        print("Pre-calculating corpus statistics (BM25 IDF, Brand Ratings)...")
        N = len(corpus)
        
        # A. BM25 IDF
        # idf = log((N - df + 0.5) / (df + 0.5) + 1)
        for term, posting_list in self.inverted_index.items():
            df_t = len(posting_list)
            self.bm25_idf[term] = np.log((N - df_t + 0.5) / (df_t + 0.5) + 1)

        # B. Brand Ratings (per Custom Score)
        # {brand: average_rating}
        brand_sums = {}
        brand_counts = {}
        
        for doc in corpus.values():
            if doc.brand and doc.average_rating:
                b = doc.brand.lower()
                brand_sums[b] = brand_sums.get(b, 0) + doc.average_rating
                brand_counts[b] = brand_counts.get(b, 0) + 1
        
        self.brand_ratings = {b: (s / brand_counts[b]) for b, s in brand_sums.items()}
        
        self._corpus_processed = True
        print("Corpus statistics ready.")


    def search(self, search_query, search_id, corpus, ranking_method):
        print(f"Search query: '{search_query}' | Method: {ranking_method}")

        # Ensure corpus stats are prepared
        if not self._corpus_processed and corpus:
            self._prepare_corpus_stats(corpus)

        results = []

        # Algorithm selection
        if ranking_method == 'tfidf':
            results = search_tfidf(search_query, self.inverted_index, self.idf_scores, corpus)
            
        elif ranking_method == 'bm25':
            results = search_bm25(search_query, self.inverted_index, self.bm25_idf, corpus)
            
        elif ranking_method == 'custom':
            results = search_custom(search_query, self.inverted_index, self.bm25_idf, corpus, self.brand_ratings)
            
        elif ranking_method == 'word2vec':
            results = search_word2vec(search_query, self.w2v_model, corpus)
            
        elif ranking_method == 'doc2vec':
            results = search_doc2vec(search_query, self.d2v_model, corpus)
            
        else:
            # Fallback or Dummy
            print(f"Unknown method {ranking_method}, using random dummy.")
            # Dummy logic
            import random
            doc_ids = list(corpus.keys())
            docs_to_return = np.random.choice(doc_ids, size=min(10, len(doc_ids)), replace=False)
            for doc_id in docs_to_return:
                doc = corpus[doc_id].model_copy(update={'score': round(random.random(), 4)})
                results.append(doc)

        return results