import pandas as pd
import numpy as np
import ast
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re, string
from myapp.search.objects import Document


# --------------------------------------------------
# Inspired in Part 2 or 3 tokenizer
# --------------------------------------------------
def preprocess_text(text):
    """
    Preprocess textual fields exactly like in Part 2 & Part 3:
    - lowercase
    - remove punctuation & digits
    - tokenize
    - remove stopwords
    - stem words
    - normalize colors
    """
    if not isinstance(text, str):
        return []

    # Lowercase
    text = text.lower()

    # Replace hyphens
    text = text.replace('-', ' ')

    # Remove punctuation and numbers
    text = re.sub(f"[{re.escape(string.punctuation)}0-9]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Load tools (NLP preprocessing)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    stop_words.update(['made', 'wear', 'comfort', 'quality', 
                       'look', 'perfect', 'style', 'great', 'cool'])
    stop_words.discard('no')
    stop_words.discard('not')

    # Filter tokens
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    # Stem
    tokens = [stemmer.stem(t) for t in tokens]

    # Color normalization
    color_map = {
        'navy': 'blue', 'grey': 'gray', 'fucsia': 'pink', 'burgundy': 'red',
        'violet': 'purple', 'beige': 'brown', 'magenta': 'pink', 'indigo': 'blue',
        'charcoal': 'gray', 'crimson': 'red', 'teal': 'green', 'lavender': 'purple',
        'mustard': 'yellow', 'turquoise': 'blue', 'peach': 'orange'
    }
    tokens = [color_map.get(t, t) for t in tokens]

    return tokens


# --------------------------------------------------
# Helper: Safely restore lists/dicts from strings
# --------------------------------------------------
def safe_literal_eval(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed == "":
            return None
        try:
            return ast.literal_eval(trimmed)
        except (ValueError, SyntaxError):
            return None
    return value


# --------------------------------------------------
# Helper: Convert numeric list → numpy array
# --------------------------------------------------
def to_numpy_array(value):
    if value is None:
        return None
    if isinstance(value, list):
        return np.array(value, dtype=float)
    if isinstance(value, str):
        try:
            lst = ast.literal_eval(value)
            return np.array(lst, dtype=float)
        except:
            return None
    return None


def ensure_list(value):
    if isinstance(value, list):
        return value
    parsed = safe_literal_eval(value)
    if isinstance(parsed, list):
        return parsed
    return []


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


# --------------------------------------------------
# BUILD CORPUS FROM DATA FILE
# --------------------------------------------------
def load_corpus(path):
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == ".json":
        print(f"[Corpus] Loading JSON from: {path}")
        df = pd.read_json(path)
    else:
        print(f"[Corpus] Loading CSV from: {path}")
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")

    # Restore fields from string → Python objects
    for col in ["processed_text", "attributes", "tokens"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_literal_eval)

    if "images" in df.columns:
        df["images"] = df["images"].apply(ensure_list)

    # TF-IDF vector (dict stored as string)
    if "tfidf_vector" in df.columns:
        df["tfidf_vector"] = df["tfidf_vector"].apply(safe_literal_eval)

    # Word2Vec vector
    if "document_vector" in df.columns:
        df["document_vector"] = df["document_vector"].apply(to_numpy_array)

    # Doc2Vec vector
    if "doc2vec_vector" in df.columns:
        df["doc2vec_vector"] = df["doc2vec_vector"].apply(to_numpy_array)

    if "out_of_stock" in df.columns:
        df["out_of_stock"] = df["out_of_stock"].apply(to_bool)

    # --------------------------------------------------
    # BUILD THE CORPUS
    # --------------------------------------------------
    corpus = {}

    for _, row in df.iterrows():

        doc = Document(
            _id=str(row.get("_id", row.get("pid"))),
            pid=str(row.get("pid")),
            title=row.get("title"),
            description=row.get("description"),

            brand=row.get("brand_facet") or row.get("brand"),
            category=row.get("category"),
            sub_category=row.get("sub_category"),
            product_details=row.get("product_details"),
            seller=row.get("seller"),
            out_of_stock=bool(row.get("out_of_stock", False)),

            selling_price=row.get("selling_price"),
            discount=row.get("discount"),
            actual_price=row.get("actual_price"),
            average_rating=row.get("average_rating"),
            url=row.get("url"),
            images=row.get("images"),

            # ADDED fields for ranking
            tokens=row.get("tokens") or [],
            processed_text=row.get("processed_text"),
            attributes=row.get("attributes"),

            tfidf_vector=row.get("tfidf_vector"),
            doc_length=row.get("doc_length"),

            document_vector=row.get("document_vector"),
            doc2vec_vector=row.get("doc2vec_vector"),

            score=None  # ranking score added later
        )

        # ---------------------------------------------
        # REBUILD TOKENS EXACTLY AS IN PART 2 & 3
        # ---------------------------------------------

        # 1. Processed text from description
        text = doc.description if doc.description else ""
        doc.processed_text = preprocess_text(text)

        # 2. Process attributes (if product_details exists)
        if isinstance(doc.product_details, dict):
            attr_text = " ".join([f"{k} {v}" for k, v in doc.product_details.items()])
            doc.attributes = preprocess_text(attr_text)
        else:
            doc.attributes = []

        # 3. Combine to final tokens list
        doc.tokens = doc.processed_text + doc.attributes
        doc.doc_length = len(doc.tokens)

        corpus[doc.pid] = doc

    print(f"[Corpus] Loaded {len(corpus)} documents.")
    return corpus
