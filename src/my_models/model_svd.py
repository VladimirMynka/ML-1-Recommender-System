from pathlib import Path
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder

from src.utils import read_files
from src.config import config
from src.my_models.model import Model


class Model_SVD(Model):
    def __init__(self):
        super().__init__()
        self.n_movies: int = 0
        self.n_users: int = 0
        self.users_le: Optional[LabelEncoder] = None
        self.movies_le: Optional[LabelEncoder] = None
        self.users_means: Optional[np.ndarray] = None
        self.vt: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.u: Optional[np.ndarray] = None
        self.df_users, self.df_movies = read_files(["users", "movies"], [None, None])

    def train(
        self,
        train_path: Path = None,
        d: int = 35
    ) -> None:
        """
        Receives the dataset filename. Performs model training. Saves the artifacts to ./model/. Logs the results.
        :param train_path: path to train dataset
        :param d: shape of the middle matrix in svd
        """
        data = read_files(["train"], [train_path])
        matrix = self._prepare_matrix(data["train"])
        self.u, self.s, self.vt = svds(matrix, k=d)
        self.s = np.diag(self.s)

    def _prepare_matrix(self, train: pd.DataFrame) -> np.ndarray:
        self._create_encoders(train)
        matrix = np.full(shape=(self.n_users, self.n_movies), fill_value=np.nan)
        for row in tqdm(train):
            matrix[row.user_id, row.movie_id] = row.rating
        self.users_means = np.nanmean(matrix, axis=1).reshape(-1, 1)
        matrix /= self.users_means
        matrix -= 1
        matrix[np.isnan(matrix)] = 0
        return matrix

    def _create_encoders(self, train: pd.DataFrame) -> None:
        self.movies_le = LabelEncoder()
        self.users_le = LabelEncoder()
        train['movie_id'] = self.movies_le.fit_transform(train['movie_id'])
        train['user_id'] = self.users_le.fit_transform(train['user_id'])
        self.n_users = len(self.users_le.classes_)
        self.n_movies = len(self.movies_le.classes_)

    def evaluate(self, data_path: Path) -> None:
        """
        Receives the dataset filename. Loads the model from ./data/model/.

        Evaluates the model with the provided dataset, prints the results and saves it to the log.
        :param data_path: path to evaluating data
        """
        pass

    def predict(self, data: list, top_m: int) -> list:
        """
        Get recommend movies for one user

        :param data: list `[[movie_id_1, movie_id_2, .., movie_id_N ], [rating_1, rating_2, .., rating_N]]` for one user
        :param top_m: how much recommended movies must be got

        :return: recommended movies with estimated rating in the same as input format
        """
        user_norm_ratings, user_mean = self._prepare_one_user(data)  # (movies_count, )
        users_to_movies = self.u @ self.s @ self.vt  # (users_count, movies_count)

        normalizer = np.sqrt((users_to_movies ** 2).sum(axis=1))

        # (users_count, movies_count) @ (movies_count, 1) = (users_count, 1)
        users_sims = users_to_movies @ user_norm_ratings.reshape((-1, 1)) / normalizer

        predicted = (users_sims * user_norm_ratings).sum(axis=0) / users_sims.sum()
        ids = np.argsort(predicted)[::-1]
        np.delete(ids, self.movies_le.transform(data[0]))

    def _prepare_one_user(self, data: list) -> (np.ndarray, float):
        user_ratings = np.full(self.n_movies, fill_value=np.nan)
        movie_ids = self.movies_le.transform(data[0])
        for movie_id, rating in zip(movie_ids, data[1]):
            user_ratings[movie_id] = rating
        user_mean = np.nanmean(user_ratings)
        user_ratings /= user_mean
        user_ratings -= 1
        return user_ratings, user_mean

    def warmup(self, path: Optional[str | Path] = None) -> None:
        """
        Loads the model from ./data/model/. Refresh if it is already loaded.
        :param path: path for loading model. Default is config['model']
        """
        self.u = np.load(path / "u.np")
        self.s = np.load(path / "s.np")
        self.vt = np.load(path / "vt.np")
        self.users_means = np.load(path / "users_means.np")

        self.users_le = LabelEncoder()
        self.movies_le = LabelEncoder()
        self.users_le.fit(np.load(path / "users_le.np"))
        self.movies_le.fit(np.load(path / "movies_le.np"))

        self.n_users = len(self.users_le.classes_)
        self.n_movies = len(self.movies_le.classes_)

    def find_similar(self, movie_id: int, N: int = 5) -> list:
        """
        Returns N (parameter, default=5) most similar movies for input movie_id
        :param movie_id: id of a movie for which must be found similar some
        :param N: count of values will be returned

        :return: Returns list [
        [movie_id_1, movie_id_2, .., movie_id_N],
        [movie_name_1, movie_name_2, .., movie_name_N]
        ] Descending sorting by similarity
        """
        movie_id = self.movies_le.transform(movie_id)
        movies_to_movies = self.vt.T @ self.vt  # (M x d) @ (d x M) = M x M
        indexes = movies_to_movies[movie_id].argsort()[::-1]

        np.delete(indexes, movie_id)

        old_indexes = self.movies_le.inverse_transform(indexes)
        old_indexes = old_indexes[:N]
        names = self._get_movies_names(old_indexes)

        return [
            old_indexes,
            names
        ]

    def _get_movies_names(self, movie_old_ids):
        return self.df_movies[np.isin(self.df_movies.movie_id, movie_old_ids)]

    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save model weights to data/model
        :param path: path for saving model. Default is config['model']
        """
        if path is None:
            path = config.model
        path = Path(path)
        np.save(str(path / "u.np"), self.u)
        np.save(str(path / "s.np"), self.s)
        np.save(str(path / "vt.np"), self.vt)
        np.save(str(path / "users_means.np"), self.users_means)
        np.save(str(path / "users_le.np"), self.users_le.classes_)
        np.save(str(path / "movies_le.np"), self.movies_le.classes_)
