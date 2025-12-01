import numpy as np
from collections import Counter
from numpy.linalg import norm
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string

# --- Preprocessing functions to preprocess queries ---

def setup_preprocessing():
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    stop_words.update(['made', 'wear', 'comfort', 'quality', 
                       'look', 'perfect', 'style', 'great', 'cool'])
    stop_words.discard('no')
    stop_words.discard('not')
    return stemmer, stop_words

STEMMER, STOP_WORDS = setup_preprocessing()

def preprocess_query(text):
    """
    Same logic used in Part 2/3 Notebooks to process the query text.
    """
    if not isinstance(text, str):
        return []
    
    text = text.replace('-', ' ')
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)
    
    tokens = word_tokenize(text)
    tokens = [STEMMER.stem(w) for w in tokens if w.isalpha() and w not in STOP_WORDS]
    
    color_map = {
        'navy': 'blue', 'grey': 'gray', 'fucsia': 'pink', 'burgundy': 'red', 
        'violet': 'purple', 'beige': 'brown', 'magenta': 'pink', 'indigo': 'blue', 
        'charcoal': 'gray', 'crimson': 'red', 'teal': 'green', 'lavender': 'purple', 
        'mustard': 'yellow', 'turquoise': 'blue', 'peach': 'orange'
    }
    tokens = [color_map.get(w, w) for w in tokens]
    
    return tokens


# --- 1. TF-IDF ---

def search_tfidf(query, inverted_index, idf_scores, corpus, k=20):
    query_tokens = preprocess_query(query)
    if not query_tokens: return []

    # 1. Candidates (AND logic)
    doc_sets = [set(inverted_index.get(t, [])) for t in query_tokens if t in inverted_index]
    if not doc_sets or len(doc_sets) < len(query_tokens): # Strict AND
        return []
    
    candidate_pids = set.intersection(*doc_sets)
    if not candidate_pids: return []

    scores = {}
    
    # Query Vector
    query_tf = Counter(query_tokens)
    query_vec = {t: (1 + np.log(tf)) * idf_scores.get(t, 0) for t, tf in query_tf.items()}
    query_norm = norm(list(query_vec.values()))

    for pid in candidate_pids:
        doc = corpus[pid]
        # Using precomputed tokens
        doc_tokens = doc.tokens 
        doc_tf = Counter(doc_tokens)
        
        dot_product = 0.0

        for term, q_weight in query_vec.items():
            if term in doc_tf:
                d_weight = (1 + np.log(doc_tf[term])) * idf_scores.get(term, 0)
                dot_product += q_weight * d_weight
        
        # Normalization
        doc_len = len(doc_tokens)
        if doc_len > 0 and query_norm > 0:
            scores[pid] = dot_product / (query_norm * np.sqrt(doc_len))
        else:
            scores[pid] = 0.0

    return _format_results(scores, corpus, k)


# --- 2. BM25 ---

def search_bm25(query, inverted_index, bm25_idf, corpus, k=20, k1=1.5, b=0.75):
    query_tokens = preprocess_query(query)
    if not query_tokens: return []

    doc_sets = [set(inverted_index.get(t, [])) for t in query_tokens if t in inverted_index]
    if not doc_sets or len(doc_sets) < len(query_tokens): return []
    candidate_pids = set.intersection(*doc_sets)
    
    scores = {}
   
    total_len = sum(len(d.tokens) for d in corpus.values())
    L_ave = (total_len / len(corpus)) if (len(corpus) > 0 and total_len > 0) else 1.0

    for pid in candidate_pids:
        doc = corpus[pid]
        doc_tokens = doc.tokens
        Ld = len(doc_tokens)
        tf_doc = Counter(doc_tokens)
        
        score = 0.0
        for t in query_tokens:
            if t not in bm25_idf: continue
            
            tf_td = tf_doc.get(t, 0)
            idf = bm25_idf[t]
            
            denom = tf_td + k1 * ((1 - b) + b * (Ld / L_ave))
            term_score = idf * ((k1 + 1) * tf_td / denom)
            score += term_score
        
        scores[pid] = score

    return _format_results(scores, corpus, k)


# --- 3. CUSTOM SCORE ---

def search_custom(query, inverted_index, bm25_idf, corpus, brand_ratings, k=20):
    """
    BM25 base + Boosts (Rating, Discount, Brand)
    """
    query_tokens = preprocess_query(query)
    if not query_tokens: return []

    doc_sets = [set(inverted_index.get(t, [])) for t in query_tokens if t in inverted_index]
    if not doc_sets or len(doc_sets) < len(query_tokens): return []
    candidate_pids = set.intersection(*doc_sets)
    
    scores = {}
    total_len = sum(len(d.tokens) for d in corpus.values())
    L_ave = (total_len / len(corpus)) if (len(corpus) > 0 and total_len > 0) else 1.0
    k1, b = 1.5, 0.75
    
    # Weights for boosts
    w_rating = 0.1
    w_discount = 0.05
    w_brand = 0.05
    
    for pid in candidate_pids:
        doc = corpus[pid]
        doc_tokens = doc.tokens
        Ld = len(doc_tokens)
        tf_doc = Counter(doc_tokens)
        
        # Base BM25
        bm25_score = 0.0
        for t in query_tokens:
            if t not in bm25_idf: continue
            tf_td = tf_doc.get(t, 0)
            idf = bm25_idf[t]
            denom = tf_td + k1 * ((1 - b) + b * (Ld / L_ave))
            bm25_score += idf * ((k1 + 1) * tf_td / denom)
            
        if bm25_score <= 0: continue

        # Boosts
        rating = doc.average_rating if doc.average_rating else 0.0
        discount = doc.discount if doc.discount else 0.0
        brand = doc.brand.lower() if doc.brand else "unknown"
        
        rating_boost = 1 + (w_rating * np.log1p(rating))
        discount_boost = 1 + (w_discount * np.log1p(discount))
        avg_brand_rating = brand_ratings.get(brand, 2.5)
        brand_boost = 1 + (w_brand * np.log1p(avg_brand_rating))
        
        scores[pid] = bm25_score * rating_boost * discount_boost * brand_boost

    return _format_results(scores, corpus, k)


# --- 4. WORD2VEC ---

def text_to_vector(text, model):
    tokens = preprocess_query(text)
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

def search_word2vec(query, model, corpus, k=20):
    if not model: return []
    query_vec = text_to_vector(query, model)
    query_norm = norm(query_vec)
    if query_norm == 0: return []

    scores = {}

    for pid, doc in corpus.items():

        vectors = [model.wv[t] for t in doc.tokens if t in model.wv]
        if not vectors: continue
        doc_vec = np.mean(vectors, axis=0)
        
        doc_norm = norm(doc_vec)
        if doc_norm > 0:
            scores[pid] = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)

    return _format_results(scores, corpus, k)

# --- 5. DOC2VEC ---

def search_doc2vec(query, model, corpus, k=20):
    if not model: return []
    tokens = preprocess_query(query)
    query_vec = model.infer_vector(tokens)
    query_norm = norm(query_vec)
    if query_norm == 0: return []
    
    scores = {}
    for pid, doc in corpus.items():
        try:
            doc_vec = model.dv[pid]
            doc_norm = norm(doc_vec)
            if doc_norm > 0:
                scores[pid] = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
        except KeyError:
            continue

    return _format_results(scores, corpus, k)


# --- HELPER ---

def _format_results(scores, corpus, k):
    top_hits = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    results = []
    for pid, score in top_hits:
        if pid in corpus:
            doc = corpus[pid].model_copy()
            doc.score = score
            results.append(doc)
    return results

