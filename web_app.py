import os
from json import JSONEncoder

import httpagentparser  # for getting the user agent as json
from flask import Flask, render_template, session
from flask import request

from myapp.analytics.analytics_data import AnalyticsData, ClickedDoc
from myapp.search.load_corpus import load_corpus
from myapp.search.objects import Document, StatsDocument
from myapp.search.search_engine import SearchEngine
from myapp.generation.rag import RAGGenerator
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env


# *** for using method to_json in objects ***
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default
# end lines ***for using method to_json in objects ***


# instantiate the Flask application
app = Flask(__name__)

# random 'secret_key' is used for persisting data in secure cookie
app.secret_key = os.getenv("SECRET_KEY")
# open browser dev tool to see the cookies
app.session_cookie_name = os.getenv("SESSION_COOKIE_NAME")
# instantiate our search engine
search_engine = SearchEngine()
# instantiate our in memory persistence
analytics_data = AnalyticsData()
# instantiate RAG generator
rag_generator = RAGGenerator()

# load documents corpus into memory.
full_path = os.path.realpath(__file__)
path, filename = os.path.split(full_path)
file_path = path + "/" + os.getenv("DATA_FILE_PATH")
corpus = load_corpus(file_path)
# Log first element of corpus to verify it loaded correctly:
print("\nCorpus is loaded... \n First element:\n", list(corpus.values())[0])


# Home URL "/"
@app.route('/')
def index():
    print("starting home url /...")

    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex() # Generate unique session id

    analytics_data.start_session(
        session['session_id'], 
        request.headers.get('User-Agent'), 
        request.remote_addr
    )

    # flask server creates a session by persisting a cookie in the user's browser.
    # the 'session' object keeps data between multiple requests. Example:
    session['some_var'] = "Some value that is kept in session"

    user_agent = request.headers.get('User-Agent')
    print("Raw user browser:", user_agent)

    user_ip = request.remote_addr
    agent = httpagentparser.detect(user_agent)

    print("Remote IP: {} - JSON user browser {}".format(user_ip, agent))
    print(session)
    return render_template('index.html', page_title="Welcome")


@app.route('/search', methods=['GET'])
def search_form_post():
    
    search_query = request.args.get('search-query')

    #defaults to TF-IDF if no ranking method is selected
    ranking_method = request.args.get('ranking-method', 'tfidf')

    if not search_query:
        print("Empty search query received...")
        return render_template('index.html', page_title="Home")

    session['last_search_query'] = search_query
    session['last_ranking_method'] = ranking_method

    results = search_engine.search(search_query, None, corpus, ranking_method=ranking_method)
    
    if 'session_id' in session:
        analytics_data.log_search(
            session['session_id'],
            search_query,
            ranking_method,
            len(results)
        )
    else:
        # Security measure: in case session_id is missing, we log a warning
        print("Warning: Search performed without session_id")

    # generate RAG response based on user query and retrieved results
    rag_response = rag_generator.generate_response(search_query, results)
    print("RAG response:", rag_response)

    found_count = len(results)
    session['last_found_count'] = found_count

    print(session)

    return render_template('results.html', results_list=results, page_title="Results", found_counter=found_count, rag_response=rag_response, ranking_method=ranking_method)


@app.route('/doc_details', methods=['GET'])
def doc_details():
    """
    Show document details page
    ### Replace with your custom logic ###
    """

    # getting request parameters:
    # user = request.args.get('user')
    print("doc details session: ", session)

    res = session.get("some_var")

    if res:
        print("recovered var from session:", res)

    # get the query string parameters from request
    clicked_doc_id = request.args.get("pid") or request.form.get("id")
    if not clicked_doc_id:
        return render_template("doc_details.html", doc=None, page_title="Document not found")
    
    print(f"click in id={clicked_doc_id}")

    rank = request.args.get('rank', 1) # Retrieve rank from query parameters

    if 'session_id' in session:
        analytics_data.log_click(session['session_id'], clicked_doc_id, rank)
    else:
        print("Warning: Click without session (user might have cookies disabled)")

    
    if clicked_doc_id not in corpus: 
        return render_template("doc_details.html", doc=None, page_title="Document not found")

    row: Document = corpus[clicked_doc_id]
    
    doc = {
        "pid": row.pid,
        "title": row.title,
        "description": row.description,
        "url": row.url or "",
        "brand": getattr(row, 'brand', '') or getattr(row, 'brand_facet', ''), # Més robust
        "selling_price": getattr(row, 'selling_price', 'N/A'),
        "average_rating": getattr(row, 'average_rating', 'N/A'),
        "discount": getattr(row, 'discount', 'N/A'),
        "category": getattr(row, 'category', ''),
        "sub_category": getattr(row, 'sub_category', ''),
        "out_of_stock": getattr(row, 'out_of_stock', False),
        "images": getattr(row, 'images', []),
        "product_details": getattr(row, 'product_details', {}),
    }

    return render_template('doc_details.html', doc=doc, page_title=row.title)


@app.route('/stats', methods=['GET'])
def stats():
    """
    Show simple statistics example. ### Replace with yourdashboard ###
    :return:
    """

    docs = []
    for doc_id in analytics_data.fact_clicks:
        row: Document = corpus[doc_id]
        count = analytics_data.fact_clicks[doc_id]
        doc = StatsDocument(pid=row.pid, title=row.title, description=row.description, url=row.url, count=count)
        docs.append(doc)
    
    # simulate sort by ranking
    docs.sort(key=lambda doc: doc.count, reverse=True)
    return render_template('stats.html', clicks_data=docs)


@app.route('/dashboard', methods=['GET'])
def dashboard():

    charts_html = analytics_data.plot_number_of_views()

    visited_docs = []
    for doc_id in analytics_data.fact_clicks.keys():
        d: Document = corpus[doc_id]
        doc = ClickedDoc(doc_id, d.description, analytics_data.fact_clicks[doc_id])
        visited_docs.append(doc)

    # simulate sort by ranking
    visited_docs.sort(key=lambda doc: doc.counter, reverse=True)

    return render_template('dashboard.html', visited_docs=visited_docs, charts_html=charts_html)


# New route added for generating an examples of basic Altair plot (used for dashboard)
@app.route('/plot_number_of_views', methods=['GET'])
def plot_number_of_views():
    return analytics_data.plot_number_of_views()


if __name__ == "__main__":
    app.run(port=8088, host="0.0.0.0", threaded=False, debug=os.getenv("DEBUG"))