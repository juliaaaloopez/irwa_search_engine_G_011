import json
import random
import altair as alt
import pandas as pd
import time
from datetime import datetime     
import httpagentparser


class AnalyticsData:
    def __init__(self):
        # In-memory tables
        self.fact_clicks = dict()
        self.searches = []  
        self.sessions = {} 
        self.requests = [] 
        self.clicks = []   
        self.last_click_by_session = {}
    
    def start_session(self, session_id, user_agent, ip):
        if session_id not in self.sessions:
            parsed = httpagentparser.detect(user_agent or "")
            browser = parsed.get("browser", {}).get("name", "Unknown")
            os_name = parsed.get("os", {}).get("name", "Unknown")
            device = "mobile" if "Mobile" in user_agent else "desktop"
            timestamp = datetime.now()

            self.sessions[session_id] = {
                "user_agent": user_agent,
                "ip": ip,
                "start_time": timestamp,
                "browser": browser,         
                "os": os_name,              
                "device": device,           
            }
            print(f"[Analytics] Session started: {session_id} | {browser} on {os_name}")
    
    def log_search(self, session_id, query, ranking_method, results_count):
        n_terms = len(query.split()) 
        timestamp = datetime.now()

        self.requests.append({
            "session_id": session_id,
            "query": query,
            "n_terms": n_terms, 
            "ranking_method": ranking_method,
            "results_count": results_count,
            "timestamp": timestamp,  
        })
        print(f"[Analytics] Search logged. Total searches now: {len(self.requests)}")


    def log_click(self, session_id, doc_id, rank):
        timestamp = datetime.now()
        doc_id = str(doc_id)
        
        click_record = {
            "session_id": session_id,
            "doc_id": doc_id,
            "rank": rank,
            "timestamp": timestamp,
            "dwell_time": None,           
        }
        self.clicks.append(click_record)

        self.last_click_by_session[session_id] = {
            "doc_id": doc_id,
            "timestamp": timestamp,
            "index": len(self.clicks) - 1, 
        }

        if doc_id in self.fact_clicks:
            self.fact_clicks[doc_id] += 1
        else:
            self.fact_clicks[doc_id] = 1

        print(f"[Analytics] Click: doc={doc_id}, rank={rank}")

    def get_clicks_df(self):
        return pd.DataFrame(self.clicks)
    
    
    def plot_number_of_views(self):
        data = [{'Document ID': doc_id, 'Number of Views': count} for doc_id, count in self.fact_clicks.items()]
        
        if not data:
            return None 

        df = pd.DataFrame(data)

        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Document ID:N', sort='-y', axis=alt.Axis(title='Document ID')),
            y=alt.Y('Number of Views:Q', axis=alt.Axis(title='Views'))
        ).properties(
            title='Number of Views per Document',
            height=250 
            # Hem tret width='container' per seguretat
        )
        
        # Utilitzem json.loads(chart.to_json()) per assegurar que les dades estan netes per a JS
        return json.loads(chart.to_json())

    def register_return_to_results(self, session_id):
        last = self.last_click_by_session.get(session_id)
        if not last:
            return  

        now = datetime.now()
        dwell_seconds = (now - last["timestamp"]).total_seconds()
        click_index = last["index"]
        
        if 0 <= click_index < len(self.clicks):
            self.clicks[click_index]["dwell_time"] = dwell_seconds

        print(f"[Analytics] Dwell time for session {session_id}, "
              f"doc={last['doc_id']}: {dwell_seconds:.2f} seconds")

   
    def plot_search_volume(self):
        if not self.requests:
            return None

        df = pd.DataFrame(self.requests)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values("timestamp")

        df_grouped = df.groupby(pd.Grouper(key='timestamp', freq='1min')).size().reset_index(name='count')

        chart = (
            alt.Chart(df_grouped)
            .mark_line(point=True)
            .encode(
                x='timestamp:T',
                y='count:Q',
                tooltip=['timestamp:T', 'count:Q'],
            )
            .properties(title="Number of Searches Over Time", height=250)
        )
        return json.loads(chart.to_json())

    def plot_query_length_distribution(self):
        if not self.requests:
            return None

        df = pd.DataFrame(self.requests)

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X('n_terms:O', title='Number of Terms in a Query', sort=list(range(1, 11))),
                y=alt.Y('count()', title='Frequency'),
                tooltip=['n_terms:Q', 'count()']
            )
            .properties(title="Distribution of Query Lengths", height=250)
        )

        return json.loads(chart.to_json())

    def plot_average_dwell_time(self):
        if not self.clicks:
            return None

        df = pd.DataFrame(self.clicks)
        df = df[df['dwell_time'].notnull()]

        if df.empty:
            return None

        df_avg = df.groupby('doc_id')['dwell_time'].mean().reset_index()

        chart = (
            alt.Chart(df_avg)
            .mark_bar()
            .encode(
                x=alt.X('doc_id:N', sort='-y'),
                y=alt.Y('dwell_time:Q', title='Average Dwell Time (seconds)'),
                tooltip=['doc_id:N', 'dwell_time:Q']
            )
            .properties(title="Average Dwell Time per Document", height=250)
        )

        return json.loads(chart.to_json())


class ClickedDoc:
    def __init__(self, doc_id, description, counter):
        self.doc_id = doc_id
        self.description = description
        self.counter = counter

    def to_json(self):
        return self.__dict__

    def __str__(self):
        return json.dumps(self)