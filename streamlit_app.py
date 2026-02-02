import streamlit as st
from recommender import load_data, build_item_knn, find_movie_by_title, recommend

st.set_page_config(page_title=" Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬  Movie Recommender")
st.write("Item-based collaborative filtering on MovieLens (ml-latest-small)")

@st.cache_data
def cached_load_data():
    return load_data("data")

@st.cache_resource
def cached_build_model(n_neighbors: int = 20):
    return build_item_knn("data", n_neighbors=n_neighbors)

ratings, movies = cached_load_data()

# Popular movies for a good default list
top_movie_ids = ratings["movieId"].value_counts().index[:200]
popular_titles = movies[movies["movieId"].isin(top_movie_ids)]["title"].unique().tolist()
popular_titles = sorted(popular_titles)

st.sidebar.header("Search & Options")
query = st.sidebar.text_input("Search movie title (partial)")
if query:
    matches = find_movie_by_title(query, movies)["title"].tolist()
    if matches:
        movie_choice = st.sidebar.selectbox("Matching titles", matches)
    else:
        st.sidebar.info("No matches found. Try another query or pick from popular movies.")
        movie_choice = st.sidebar.selectbox("Choose movie (popular)", popular_titles)
else:
    movie_choice = st.sidebar.selectbox("Choose movie (popular)", popular_titles)

k = st.sidebar.slider("Number of recommendations", 1, 20, 10)
recompute = st.sidebar.button("Rebuild model now")

st.markdown("---")
col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Selected movie")
    st.write(f"**{movie_choice}**")
    if st.button("Show similar movies"):
        # Show quick neighbors using a small model
        with st.spinner("Computing recommendations..."):
            model_obj = cached_build_model(max(20, k + 5))
            try:
                recs = recommend(movie_choice, k=k, model_obj=model_obj)
                for title, score in recs:
                    st.write(f"- {title} — similarity {score:.3f}")
            except Exception as e:
                st.error(str(e))

with col2:
    st.subheader("Top recommendations")
    if st.button("Get recommendations"):
        with st.spinner("Building model (first run may take ~30s)..."):
            model_obj = cached_build_model(max(20, k + 5))
        try:
            recs = recommend(movie_choice, k=k, model_obj=model_obj)
            rec_df = {"title": [r[0] for r in recs], "similarity": [r[1] for r in recs]}
            st.table(rec_df)
        except Exception as e:
            st.error(str(e))

st.markdown("---")
