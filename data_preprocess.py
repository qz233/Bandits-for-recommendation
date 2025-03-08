import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


file_path = "./ml-100k"


# Load ratings data
def load_ratings_data(file_path, range, full_users=None, full_movies=None):
    ratings_columns = ["user_id", "movie_id", "rating", "timestamp"]
    ground_truth_rating = pd.read_csv(f"{file_path}/u.data", sep="\t", names=ratings_columns)

    movie_columns = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure",
                    "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
                    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    movies = pd.read_csv(f"{file_path}/u.item", sep="|", names=movie_columns, encoding="latin-1", usecols=[0, 1])

    # Merge ratings and movies
    data = pd.merge(ground_truth_rating, movies, on="movie_id")

    min_time = data['timestamp'].min()
    max_time = data['timestamp'].max()

    lower_bound = min_time + range[0] * (max_time - min_time)
    upper_bound = min_time + range[1] * (max_time - min_time)

    subset_df = data[(data['timestamp'] >= lower_bound) & (data['timestamp'] <= upper_bound)]

    rating_matrix = subset_df.pivot(index='user_id', columns='movie_id', values='rating')

    if full_users is not None and full_movies is not None:
        rating_matrix = rating_matrix.reindex(index=full_users, columns=full_movies, fill_value=0)

    rating_matrix = rating_matrix.fillna(0)
    return csr_matrix(rating_matrix.to_numpy()), rating_matrix.index, rating_matrix.columns


class Environment:
    def __init__(self, ground_truth_rating, initial_rating):
        self.gt_rating = ground_truth_rating
        self.prev_rating = initial_rating.copy()

    def option(self, user_id):
        gt_movies = set(self.gt_rating[user_id].indices)
        prev_movies = set(self.prev_rating[user_id].indices)

        available_movies = list(gt_movies - prev_movies)
        return sorted(available_movies)

    def update(self, user_id, movie_id):
        options = self.option(user_id)
        rating = self.gt_rating[user_id, movie_id]
        best_rating = self.gt_rating[user_id, options].max()
        new_rating = csr_matrix(([rating], ([user_id], [movie_id])),
                                shape=self.prev_rating.shape, dtype=np.float64)
        self.prev_rating += new_rating  
        #self.prev_rating[user_id, movie_id] = rating
        return rating, best_rating - rating


if __name__ == "__main__":
    ratings, full_users, full_movies = load_ratings_data(file_path, [0, 1])
    initial_ratings, _, _ = load_ratings_data(file_path, [0, 0.2], full_users, full_movies)

    print(initial_ratings.shape)
    print(ratings.shape)
