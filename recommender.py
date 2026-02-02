"""KNN-based item-item movie recommender using MovieLens ml-latest-small.

Functions:
- download_movielens(data_dir='data')
- build_item_knn(data_dir='data', n_neighbors=20)
- recommend(movie_title, k=10, model=None, data_dir='data')
- search_titles(query, movies_df)

This module downloads the dataset automatically on first use and builds a NearestNeighbors
model over the item-user rating matrix (movies x users) using cosine distance.
"""

from pathlib import Path
import zipfile
import requests
import io
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_movielens(data_dir: str = "data") -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-latest-small.zip"
    extracted_flag = data_dir / "ml-latest-small"

    if extracted_flag.exists():
        return data_dir

    # Download and extract
    r = requests.get(MOVIELENS_URL, stream=True, timeout=30)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(data_dir)

    # Touch a flag file
    extracted_flag.mkdir(exist_ok=True)
    return data_dir


def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    ratings_path = data_dir / "ml-latest-small" / "ratings.csv"
    movies_path = data_dir / "ml-latest-small" / "movies.csv"

    if not ratings_path.exists() or not movies_path.exists():
        download_movielens(data_dir)

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    return ratings, movies


def build_item_knn(data_dir: str = "data", n_neighbors: int = 20, save_path: str = None):
    """Builds and returns (model, item_user_matrix, movies_df, movieid_to_idx).
    Model uses cosine distance (so similarity = 1 - distance).
    """
    ratings, movies = load_data(data_dir)

    # Create item-user matrix
    item_user = ratings.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)

    # Fit NearestNeighbors
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(item_user.values)

    movieid_to_idx = {movieId: idx for idx, movieId in enumerate(item_user.index)}
    idx_to_movieid = {idx: movieId for movieId, idx in movieid_to_idx.items()}

    result = {
        "model": model,
        "item_user": item_user,
        "movies": movies,
        "movieid_to_idx": movieid_to_idx,
        "idx_to_movieid": idx_to_movieid,
    }

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(result, f)

    return result


def load_saved_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_movie_by_title(query: str, movies_df: pd.DataFrame) -> pd.DataFrame:
    q = query.strip().lower()
    mask = movies_df["title"].str.lower().str.contains(q)
    return movies_df[mask]


def recommend(movie_title: str, k: int = 10, model_obj: dict = None, data_dir: str = "data") -> list[tuple[str, float]]:
    """Return list of (title, similarity) tuples for top-k recommendations for the given movie_title.
    If multiple movies match the title, uses the first match (you can use find_movie_by_title to disambiguate).
    """
    if model_obj is None:
        model_obj = build_item_knn(data_dir, n_neighbors=max(20, k + 5))

    item_user = model_obj["item_user"]
    model = model_obj["model"]
    movies = model_obj["movies"]
    movieid_to_idx = model_obj["movieid_to_idx"]
    idx_to_movieid = model_obj["idx_to_movieid"]

    # Find movieId(s) matching title exactly or approximately
    candidates = movies[movies["title"].str.lower() == movie_title.strip().lower()]
    if candidates.empty:
        # try contains
        candidates = find_movie_by_title(movie_title, movies)
        if candidates.empty:
            raise ValueError(f"Movie title '{movie_title}' not found in dataset")

    movieId = int(candidates.iloc[0]["movieId"])
    if movieId not in movieid_to_idx:
        raise ValueError("Movie found in movies.csv but has no ratings in dataset")

    idx = movieid_to_idx[movieId]
    distances, indices = model.kneighbors(item_user.values[idx].reshape(1, -1), n_neighbors=k + 1)
    distances = distances.flatten()
    indices = indices.flatten()

    results = []
    for dist, idx in zip(distances[1:], indices[1:]):  # skip the first (itself)
        mid = idx_to_movieid[idx]
        title = movies[movies["movieId"] == mid].iloc[0]["title"]
        similarity = 1 - dist
        results.append((title, float(similarity)))
    return results


if __name__ == "__main__":
    # Quick local test
    model_obj = build_item_knn()
    recs = recommend("Toy Story (1995)", k=5, model_obj=model_obj)
    for title, score in recs:
        print(f"{title} ({score:.3f})")
