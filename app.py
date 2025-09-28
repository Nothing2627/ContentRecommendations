import streamlit as st
import pandas as pd
import joblib
import base64

# ---------- GLOBAL STYLES ----------
st.set_page_config(page_title="Get Recommendations", layout="wide")

# Load local image and convert to base64
def get_base64_of_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

local_bg = get_base64_of_image(r"nothing.jpeg")

# Inject CSS for attractive styling
st.markdown(f"""
    <style>
    /* Background with gradient overlay */
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{local_bg}") no-repeat center center fixed;
        background-size: cover;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(0,0,0,0.75), rgba(30,30,30,0.75));
        z-index: -1;
    }}

    /* Headings */
    h1, h2, h3, h4 {{
        font-family: 'Georgia', serif;
        font-weight: 700;
        color: #f5f5f5 !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
    }}

    /* Recommendation cards */
    .recommend-box {{
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    .recommend-box:hover {{
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.5);
        background: rgba(255, 255, 255, 0.15);
    }}

    /* Buttons */
    div.stButton > button {{
        background: linear-gradient(135deg, #ff4d4d, #ff9966);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        transition: all 0.2s ease;
    }}
    div.stButton > button:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0,0,0,0.5);
    }}

    /* Input fields */
    .stTextInput input {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        color: #fff;
        border: 1px solid rgba(255,255,255,0.3);
    }}
    .stSelectbox div[data-baseweb="select"] {{
        background: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
    }}

    /* Links */
    a {{
        color: #ffcc70;
        text-decoration: none;
        font-weight: bold;
    }}
    a:hover {{
        color: #ffdcae;
        text-decoration: underline;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- LOAD MODELS & DATA ----------
books = pd.read_pickle("books.pkl")
book_model = joblib.load("book_similarity_nn.pkl")
book_vectorizer = joblib.load("book_tfidf_vectorizer.pkl")

movies = pd.read_pickle("movies.pkl")
movie_similarity = joblib.load("movie_similarity.pkl")

songs = pd.read_pickle("songs.pkl")
song_model = joblib.load("songs_nn.pkl")
song_scaler = joblib.load("songs_scaler.pkl")

articles = pd.read_pickle("clean_articles.pkl")
article_model = joblib.load("article_similarity_nn.pkl")
article_vectorizer = joblib.load("article_tfidf_vectorizer.pkl")

blogs = pd.read_pickle("clean_blogs.pkl")
blog_model = joblib.load("blog_similarity_nn.pkl")
blog_vectorizer = joblib.load("blog_tfidf_vectorizer.pkl")

news = pd.read_pickle("clean_news.pkl")
news_model = joblib.load("news_similarity_nn.pkl")
news_vectorizer = joblib.load("news_tfidf_vectorizer.pkl")

# ---------- FUNCTIONS ----------
def recommend_books(title, top_n=5):
    query_vec = book_vectorizer.transform([title])
    distances, indices = book_model.kneighbors(query_vec, n_neighbors=top_n+1)
    return books.iloc[indices[0]]["book_title"].values[1:]

def recommend_movies(title, top_n=5):
    idx = movies[movies['title'] == title].index[0]
    distances = movie_similarity[idx]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
    return [movies.iloc[i[0]].title for i in movie_list]

def recommend_songs(track_name, top_n=5):
    matches = songs[songs['track_name'].str.lower() == track_name.lower()]
    if matches.empty:
        return ["Song not found."]
    song_index = matches.index[0]
    song_features = songs.loc[song_index, audio_features].values.reshape(1, -1)
    scaled_features = song_scaler.transform(song_features)
    distances, indices = song_model.kneighbors(scaled_features, n_neighbors=top_n+1)
    recs = []
    for idx in indices[0]:
        if idx != song_index:
            recs.append(f"{songs.iloc[idx]['track_name']} - {songs.iloc[idx]['track_artist']}")
    return recs

audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

def recommend_articles(title, top_n=5):
    query_vec = article_vectorizer.transform([title])
    distances, indices = article_model.kneighbors(query_vec, n_neighbors=top_n+1)
    return articles.iloc[indices[0]][["title", "url"]].iloc[1:]

def recommend_blogs(title, top_n=5):
    query_vec = blog_vectorizer.transform([title])
    distances, indices = blog_model.kneighbors(query_vec, n_neighbors=top_n+1)
    return blogs.iloc[indices[0]]["title"].values[1:]

def recommend_news(title, top_n=5):
    query_vec = news_vectorizer.transform([title])
    distances, indices = news_model.kneighbors(query_vec, n_neighbors=top_n+1)
    return news.iloc[indices[0]][["title", "url"]].iloc[1:]

# ---------- HEADER ----------
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("<h1 style='margin:0;'>🎬 MovieBuddy</h1>", unsafe_allow_html=True)
with col2:
    search, profile = st.columns([5, 1])
    with search:
        query = st.text_input("🔍 Search for anything", label_visibility="collapsed")
    with profile:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=40)

st.write("---")

# ---------- MAIN LAYOUT ----------
left, right = st.columns([2, 3])

with left:
    st.markdown("<h2 style='font-size:52px; line-height:1.2;'>Books &<br>Movies</h2>", unsafe_allow_html=True)
    st.write("💡 Discover personalized recommendations for books, movies, songs, articles, blogs, and news tailored to your taste.")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📖 Recommend Books (Quick)"):
            st.session_state["tab"] = "Books"
    with col_btn2:
        if st.button("🎬 Recommend Movies (Quick)"):
            st.session_state["tab"] = "Movies"

with right:
    c1, c2 = st.columns(2)
    with c1:
        st.image("https://upload.wikimedia.org/wikipedia/en/f/f9/Spider-Man_Homecoming_poster.jpg", width=180, caption="Spider-Man: Homecoming")
        st.image("https://m.media-amazon.com/images/I/81ai6zx6eXL._AC_SY679_.jpg", width=180, caption="Avengers: Endgame")
    with c2:
        st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fmarvelcinematicuniverse.fandom.com%2Fwiki%2FAnt-Man&psig=AOvVaw2Rn9f7nfjab69YMc7PiGW-&ust=1759054175623000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCLDfgMzZ-I8DFQAAAAAdAAAAABAX", width=180, caption="Ant-Man")
        st.image("https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.rottentomatoes.com%2Fm%2Fi_am_number_four&psig=AOvVaw3u4o5vvJMQQwgt4IOJ7mwe&ust=1759054220147000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCPCJmeLZ-I8DFQAAAAAdAAAAABAE", width=180, caption="I Am Number Four")

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["📚 Books", "🎬 Movies", "🎵 Songs", "📰 Articles", "📝 Blogs", "🗞 News"]
)

with tab1:
    st.header("📚 Book Recommendations")
    books.columns = books.columns.str.strip().str.lower().str.replace(" ", "_")
    if "book_title" in books.columns:
        selected = st.selectbox("Choose a Book:", books['book_title'].dropna().unique())
        if st.button("Recommend Books"):
            recs = recommend_books(selected)
            for r in recs:
                st.markdown(f"<div class='recommend-box'>{r}</div>", unsafe_allow_html=True)

with tab2:
    st.header("🎬 Movie Recommendations")
    selected = st.selectbox("Choose a Movie:", movies['title'].unique())
    if st.button("Recommend Movies"):
        recs = recommend_movies(selected)
        for r in recs:
            st.markdown(f"<div class='recommend-box'>{r}</div>", unsafe_allow_html=True)

with tab3:
    st.header("🎵 Song Recommendations")
    selected = st.selectbox("Choose a Song:", songs['track_name'].unique())
    if st.button("Recommend Songs"):
        recs = recommend_songs(selected)
        for r in recs:
            st.markdown(f"<div class='recommend-box'>{r}</div>", unsafe_allow_html=True)

with tab4:
    st.header("📰 Article Recommendations")
    selected = st.selectbox("Choose an Article:", articles['title'].unique())
    if st.button("Recommend Articles"):
        recs = recommend_articles(selected)
        for _, row in recs.iterrows():
            st.markdown(f"<div class='recommend-box'><a href='{row['url']}' target='_blank'>{row['title']}</a></div>", unsafe_allow_html=True)

with tab5:
    st.header("📝 Blog Recommendations")
    selected = st.selectbox("Choose a Blog:", blogs['title'].unique())
    if st.button("Recommend Blogs"):
        recs = recommend_blogs(selected)
        for r in recs:
            st.markdown(f"<div class='recommend-box'>{r}</div>", unsafe_allow_html=True)

with tab6:
    st.header("🗞 News Recommendations")
    selected = st.selectbox("Choose a News Article:", news['title'].unique())
    if st.button("Recommend News"):
        recs = recommend_news(selected)
        for _, row in recs.iterrows():
            st.markdown(f"<div class='recommend-box'><a href='{row['url']}' target='_blank'>{row['title']}</a></div>", unsafe_allow_html=True)
