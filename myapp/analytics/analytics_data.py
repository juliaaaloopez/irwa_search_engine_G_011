import json
import random
import altair as alt
import pandas as pd
import time


class AnalyticsData:
    """
    An in memory persistence object.
    Declare more variables to hold analytics tables.
    """
    # Example of statistics table
    # fact_clicks is a dictionary with the click counters: key = doc id | value = click counter

    ### Please add your custom tables here:
    def __init__(self):
        # In-memory tables
        self.fact_clicks = dict()
        
        self.sessions = {} # {session_id: {user_agent, ip, timestamp...}}
        self.requests = [] # [{session_id, query, timestamp, results_count...}]
        self.clicks = []   # [{session_id, doc_id, rank, timestamp, dwell_time...}]
    
    def start_session(self, session_id, user_agent, ip):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "user_agent": user_agent,
                "ip": ip,
                "start_time": time.time()
            }
    
    def log_search(self, session_id, query, ranking_method, results_count):
        self.requests.append({
            "session_id": session_id,
            "query": query,
            "ranking_method": ranking_method,
            "results_count": results_count,
            "timestamp": time.time()
        })

    def log_click(self, session_id, doc_id, rank):
        self.clicks.append({
            "session_id": session_id,
            "doc_id": doc_id,
            "rank": rank,
            "timestamp": time.time()
        })

        if doc_id in self.fact_clicks:
            self.fact_clicks[doc_id] += 1
        else:
            self.fact_clicks[doc_id] = 1

    def get_clicks_df(self):
        return pd.DataFrame(self.clicks)
    
    def plot_number_of_views(self):
        # Prepare data
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        
        # Message if no data
        if not data:
            return "<div class='alert alert-info'>There's no click data yet.</div>"

        df = pd.DataFrame(data)

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', axis=alt.Axis(title='Document ID')),
            y=alt.Y('Number of Views:Q', axis=alt.Axis(title='Vistes'))
        ).properties(
            title='Number of Views per Document',
            height=230,  
            width='container'
        )
        
        return chart.to_html()


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        """
        Print the object content as a JSON string
        """
        return json.dumps(self)
