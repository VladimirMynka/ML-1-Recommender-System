import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.config import config
from src.my_models.model import Model
from src.utils import read_files


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
        self.train_rmse = -1
        self.test_rmse = -1

        data = read_files(["users", "movies"], [None, None])
        self.df_users, self.df_movies = data["users"], data["movies"]

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
        matrix = self._create_matrix(data["train"])
        normalized_matrix = self._normalize_matrix(matrix)
        self.u, self.s, self.vt = svds(normalized_matrix, k=d)
        self.s = np.diag(self.s)

        rmse = self._calculate_rmse((matrix + 1) * self.users_means)
        self.train_rmse = rmse

        logging.info(f"Trained! Train RMSE: {rmse}")

    def _prepare_matrix(self, train: pd.DataFrame) -> np.ndarray:
        self._create_encoders(train)
        matrix = self._create_matrix(train)
        matrix = self._normalize_matrix(matrix)
        return matrix

    def _create_encoders(self, train: pd.DataFrame) -> None:
        self.movies_le = LabelEncoder()
        self.users_le = LabelEncoder()

        self.movies_le.fit(self.df_movies.movie_id)
        self.users_le.fit(self.df_users.user_id)

        train['movie_id'] = self.movies_le.transform(train['movie_id'])
        train['user_id'] = self.users_le.transform(train['user_id'])

        self.n_users = len(self.users_le.classes_)
        self.n_movies = len(self.movies_le.classes_)

    def evaluate(self, data_path: Optional[Path | str] = None, **kwargs) -> None:
        """
        Receives the dataset filename. Loads the model from ./data/model/.

        Evaluates the model with the provided dataset, prints the results and saves it to the log
        :param data_path: path to evaluating data
        """
        data = read_files(["test"], [data_path])
        val_matrix = self._create_matrix_for_evaluating(data['test'])

        rmse = self._calculate_rmse(val_matrix)
        self.test_rmse = rmse

        logging.info(f"Validation RMSE: {rmse}")

    def _calculate_rmse(self, val_matrix):
        pred_matrix = (self.u @ self.s @ self.vt)
        pred_matrix += 1
        pred_matrix *= self.users_means

        return np.sqrt(np.nanmean((val_matrix - pred_matrix) ** 2))

    def _create_matrix_for_evaluating(self, test_df: pd.DataFrame):
        test_df['user_id'] = self.users_le.transform(test_df['user_id'])
        test_df['movie_id'] = self.movies_le.transform(test_df['movie_id'])
        return self._create_matrix(test_df)

    def _create_matrix(self, dataframe: pd.DataFrame):
        matrix = np.full(shape=(self.n_users, self.n_movies), fill_value=np.nan)
        for row in tqdm(dataframe.iloc):
            matrix[row.user_id, row.movie_id] = row.rating
        return matrix

    def _normalize_matrix(self, matrix, replace_users_means: bool = True):
        matrix = matrix.copy()
        if replace_users_means:
            self.users_means = np.nanmean(matrix, axis=1).reshape(-1, 1)
        matrix /= self.users_means
        matrix -= 1
        matrix[np.isnan(matrix)] = 0
        return matrix

    def predict(self, data: list, top_m: int, **kwargs) -> list:
        """
        Get recommend movies for one user. Do it by users_to_movies matrix

        :param data: list `[[movie_id_1, movie_id_2, .., movie_id_N ], [rating_1, rating_2, .., rating_N]]` for one user
        :param top_m: how much recommended movies must be got

        :return: recommended movies with estimated rating in the same as input format
        """
        self._check_for_predict_data(data)
        logging.info(f"Predict {top_m} movies...")
        user_norm_ratings, user_mean = self._prepare_one_user(data)  # (movies_count, )
        evaluated_movies = self.movies_le.transform(data[0])

        users_to_movies = self.u @ self.s @ self.vt  # (users_count, movies_count)

        normalizer = np.sqrt((users_to_movies ** 2).sum(axis=1))[evaluated_movies]  # length of users vectors

        # (users_count, only_evaluated_movies) @ (only_evaluated_movies, 1) = (users_count, 1)
        users_sims = users_to_movies[:, evaluated_movies] \
                     @ user_norm_ratings[evaluated_movies].reshape((-1, 1)) \
                     / normalizer.reshape((-1, 1))

        predicted = (users_sims * users_to_movies).sum(axis=0) / users_sims.sum()

        return self._get_top_m_from_predicted(predicted, evaluated_movies, user_mean, top_m, data[1])

    def predict2(self, data: list, top_m: int, **kwargs) -> list:
        """
        Get recommend movies for one user. Do it by movies_to_movies matrix

        :param data: list `[[movie_id_1, movie_id_2, .., movie_id_N ], [rating_1, rating_2, .., rating_N]]` for one user
        :param top_m: how much recommended movies must be got

        :return: recommended movies with estimated rating in the same as input format
        """
        self._check_for_predict_data(data)
        logging.info(f"Predict {top_m} movies...")
        user_norm_ratings, user_mean = self._prepare_one_user(data)  # (movies_count, )
        evaluated_movies = self.movies_le.transform(data[0])

        movies_to_movies = self.vt.T @ self.vt  # (M x d) @ (d x M) = M x M

        weights = user_norm_ratings[evaluated_movies]

        predicted = (weights.reshape((-1, 1)) * movies_to_movies[evaluated_movies]).sum(axis=0)

        return self._get_top_m_from_predicted(predicted, evaluated_movies, user_mean, top_m, data[1])

    def _get_top_m_from_predicted(self, predicted, evaluated_movies, user_mean, top_m, source_marks):
        ids = np.argsort(predicted)[::-1]  # top ids
        ids = ids[~np.isin(ids, evaluated_movies)]  # drop already marked movies
        ids = ids[:top_m]  # only first top_m

        old_ids = self.movies_le.inverse_transform(ids)
        ratings = (predicted[ids] + 1) * user_mean
        ratings = self._rescale_array(ratings, min(source_marks), max(source_marks))

        logging.info("Predicted!")

        return [
            old_ids.tolist(),
            ratings.tolist()
        ]

    def _check_for_predict_data(self, data):
        if len(data) != 2:
            logging.error(f"Data is a list of two lists! Two! TWO! Your lists count: {len(data)}")
            raise ValueError(f"Data is a list of two lists! Two! TWO! Your lists count: {len(data)}")
        if len(data[0]) != len(data[1]):
            logging.error(f"Data is a list of two lists with equal length! "
                          f"Equal! EQUAL! Your lists length: {[len(data[0]), len(data[1])]}")
            raise ValueError(f"Data is a list of two lists with equal length! "
                             f"Equal! EQUAL! Your lists length: {[len(data[0]), len(data[1])]}")
        isin = np.isin(data[0], self.movies_le.classes_)
        if isin.mean() < 1:
            unknown = list(np.array(data[0])[~isin])
            logging.error(f"Unknown movie ids: {unknown}")
            raise ValueError(f"Unknown movie ids: {unknown}")

    def _prepare_one_user(self, data: list) -> (np.ndarray, float):
        user_ratings = np.full(self.n_movies, fill_value=np.nan)
        movie_ids = self.movies_le.transform(data[0])
        for movie_id, rating in zip(movie_ids, data[1]):
            user_ratings[movie_id] = rating
        user_mean = np.nanmean(user_ratings)
        user_ratings /= user_mean
        user_ratings -= 1
        user_ratings[np.isnan(user_ratings)] = 0
        return user_ratings, user_mean

    def warmup(self, path: Optional[str | Path] = None) -> None:
        """
        Loads the model from ./data/model/. Refresh if it is already loaded.
        :param path: path for loading model. Default is config['model']
        """
        if path is None:
            path = config.model
        self._check_warmup_path(path)

        logging.info(f"Download model from {path}")
        try:
            self.u = np.load(f"{path}/u.np.npy")
            self.s = np.load(f"{path}/s.np.npy")
            self.vt = np.load(f"{path}/vt.np.npy")
            self.users_means = np.load(f"{path}/users_means.np.npy")

            self.users_le = LabelEncoder()
            self.movies_le = LabelEncoder()
            self.users_le.fit(np.load(f"{path}/users_le.np.npy"))
            self.movies_le.fit(np.load(f"{path}/movies_le.np.npy"))

            self.n_users = len(self.users_le.classes_)
            self.n_movies = len(self.movies_le.classes_)
        except OSError as e:
            logging.error("Can't load model", exc_info=e)
            raise e

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
        logging.info(f"Search {N} similar movies...")
        if movie_id not in self.movies_le.classes_:
            logging.error("Unknown movie_id")

        movie_id = self.movies_le.transform([movie_id])[0]
        movies_to_movies = self.vt.T @ self.vt  # (M x d) @ (d x M) = M x M
        indexes = movies_to_movies[movie_id].argsort()[::-1]

        indexes = indexes[indexes != movie_id]

        old_indexes = self.movies_le.inverse_transform(indexes)
        old_indexes = old_indexes[:N]
        names = self.get_movies_names(old_indexes)

        logging.info(f"Searched!")
        return [
            old_indexes.tolist(),
            names.tolist()
        ]

    def get_movies_names(self, movie_old_ids):
        movies = self.df_movies[np.isin(self.df_movies.movie_id, movie_old_ids)]
        movies = movies.set_index("movie_id")
        movies = movies.loc[movie_old_ids]
        return movies.title.values

    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save model weights to data/model
        :param path: path for saving model. Default is config['model']
        """
        if path is None:
            path = config.model
        logging.info(f"Save model into {path}...")
        path = Path(path)
        path.mkdir(exist_ok=True)
        np.save(str(path / "u.np"), self.u)
        np.save(str(path / "s.np"), self.s)
        np.save(str(path / "vt.np"), self.vt)
        np.save(str(path / "users_means.np"), self.users_means)
        np.save(str(path / "users_le.np"), self.users_le.classes_)
        np.save(str(path / "movies_le.np"), self.movies_le.classes_)
