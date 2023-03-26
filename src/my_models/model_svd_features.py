from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.preprocessing import LabelEncoder

from src.utils import read_files
from src.config import config
from src.my_models.model_svd import Model_SVD


class Model_SVD_With_Features(Model_SVD):
    def __init__(self):
        super().__init__()
        self.users_features = []
        self.movies_features = []

    def train(
        self,
        train_path: Path = None,
        d: int = 35,
        **kwargs
    ) -> None:
        """
        Recieves the dataset filename. Performs model training. Saves the artifacts to ./model/. Logs the results.
        :param train_path: path to train dataset
        :param d: shape of the middle matrix in svd
        :keyword movies_path: path to movies dataset
        :keyword users_path: path to users dataset
        """
        data = self._read_data(train_path, **kwargs)
        users_dataset, movies_dataset = self._prepare_movies_and_users_datasets()

        self._create_encoders(data["train"])
        matrix = self._create_matrix(data["train"])
        normalized_matrix = self._normalize_matrix(matrix)
        self._fill_normalized_matrix_with_additional_data(normalized_matrix, users_dataset, movies_dataset)

        self.u, self.s, self.vt = svds(normalized_matrix, k=d)
        self.s = np.diag(self.s)

        rmse = self._calculate_rmse(matrix)
        self.train_rmse = rmse

        logging.info(f"Trained! Train RMSE: {rmse}")

    def _read_data(self, train_path, **kwargs):
        movies_path, users_path = None, None
        if "movies_path" in kwargs:
            movies_path = kwargs['movies_path']
        if "users_path" in kwargs:
            users_path = kwargs['users_path']
        data = read_files(["train", "movies", "users"], [train_path, movies_path, users_path])
        self.df_users = data["users"]
        self.df_movies = data["movies"]
        return data

    def _prepare_movies_and_users_datasets(self):
        users_dataset = self._prepare_additional_users_dataset()  # also cut df_users to [["user_id"]]
        movies_dataset = self._prepare_additional_movies_dataset()

        self.additional_users_encoder = LabelEncoder()
        users_features_ids = self.additional_users_encoder.fit_transform(self.users_features)

        self.additional_movies_encoder = LabelEncoder()
        movies_features_ids = self.additional_movies_encoder.fit_transform(self.movies_features)

        self.df_users = pd.concat(
            [self.df_users, pd.DataFrame({"user_id": movies_features_ids})],
            ignore_index=True
        )

        self.df_movies = pd.concat(
            [self.df_movies, pd.DataFrame({"movie_id": users_features_ids, "title": self.users_features})],
            ignore_index=True
        )
        return users_dataset, movies_dataset

    def _prepare_additional_users_dataset(self):
        dataset = pd.concat([
            self.df_users.user_id,
            pd.get_dummies(self.df_users.gender, drop_first=False, prefix='gender'),
            pd.get_dummies(self.df_users.age, drop_first=False, prefix='age'),
            pd.get_dummies(self.df_users.occupation, drop_first=False, prefix='occupation')
        ], axis=1)
        self.users_features = dataset.columns.drop("user_id")
        self.df_users = self.df_users[["user_id"]]
        return self._flatten(dataset, "user_id")

    @staticmethod
    def _map_year_to_period(year, mapper):
        key = None
        for key in mapper:
            if year <= key:
                break
        return mapper[key]

    def _prepare_additional_movies_dataset(self):
        df_movie = self.df_movies
        df_movie['year'] = df_movie.title.str.slice(-5, -1).astype(int)
        years_mapper = {
            10 * i: f"{10 * (i - 1)}-{10 * i}"
            for i
            in range(df_movie.year.min() // 10, df_movie.year.max() // 10)
        }
        df_movie['year_period'] = df_movie.year.apply(lambda elem: self._map_year_to_period(elem, years_mapper))

        dataset = pd.concat([
            df_movie.movie_id,
            pd.get_dummies(df_movie.year_period, drop_first=True, prefix='year_period'),
            pd.get_dummies(df_movie.genre, drop_first=True, prefix='genre')
        ], axis=1)
        self.movies_features = dataset.columns.drop("movie_id")
        self.df_movies = df_movie[["movie_id", "title"]]
        return self._flatten(dataset, "movie_id")

    @staticmethod
    def _flatten(df, id_column):
        dfs = []
        for i in df.columns.drop(id_column):
            df_one = df[df[i] == 1][[id_column, i]]
            df_one = df_one.rename(columns={i: "feature_name"})
            df_one["feature_name"] = i
            dfs.append(df_one)
        return pd.concat(dfs, axis=0)

    def _fill_normalized_matrix_with_additional_data(self, matrix, users_dataset, movies_dataset):
        movie_ids = self.movies_le.transform(movies_dataset["movie_id"])
        movie_features_ids = self.users_le.transform(
            self.additional_movies_encoder.transform(movies_dataset["feature_name"])
        )
        for movie_id, feature_id in zip(movie_ids, movie_features_ids):
            matrix[feature_id, movie_id] = 1

        user_ids = self.users_le.transform(users_dataset["user_id"])
        user_features_ids = self.movies_le.transform(
            self.additional_users_encoder.transform(users_dataset["feature_name"])
        )
        for user_id, feature_id in zip(user_ids, user_features_ids):
            matrix[user_id, feature_id] = 1

    def predict(self, data: list, top_m: int, **kwargs) -> list:
        """
        Get recommend movies for one user. Do it by users_to_movies matrix

        :param data: list `[[movie_id_1, movie_id_2, .., movie_id_N ], [rating_1, rating_2, .., rating_N]]` for one user
        :param top_m: how much recommended movies must be got
        :keyword is_male: bool
        :keyword age: age group of user. Transforms with config.available_age_groups
        :keyword occupation: occupation of user. Number from 1 to 20

        :return: recommended movies with estimated rating in the same as input format
        """
        user_norm_ratings, user_mean, evaluated_movies = self._prepare_to_predict(data, top_m, **kwargs)

        predicted = self._get_predictions_by_users_to_movies(evaluated_movies, user_norm_ratings)

        return self._get_top_m_from_predicted(predicted, evaluated_movies, user_mean,
                                              top_m, data[1], self.users_features)

    def predict2(self, data: list, top_m: int, **kwargs) -> list:
        """
        Get recommend movies for one user. Do it by movies_to_movies matrix

        :param data: list `[[movie_id_1, movie_id_2, .., movie_id_N ], [rating_1, rating_2, .., rating_N]]` for one user
        :param top_m: how much recommended movies must be got
        :keyword is_male: bool
        :keyword age: age group of user. Transforms with config.available_age_groups
        :keyword occupation: occupation of user. Number from 1 to 20

        :return: recommended movies with estimated rating in the same as input format
        """
        movies_to_movies = self.vt.T @ self.vt  # (M x d) @ (d x M) = M x M

        user_norm_ratings, user_mean, evaluated_movies = self._prepare_to_predict(data, top_m, **kwargs)

        weights = user_norm_ratings[evaluated_movies]
        predicted = (weights.reshape((-1, 1)) * movies_to_movies[evaluated_movies]).sum(axis=0)

        return self._get_top_m_from_predicted(predicted, evaluated_movies, user_mean,
                                              top_m, data[1], self.users_features)

    def _prepare_to_predict(self, data: list, top_m: int, **kwargs):
        self._check_for_predict_data(data)

        additional_data = self._extract_additional_data(kwargs)

        logging.info(f"Predict {top_m} movies...")
        user_norm_ratings, user_mean = self._prepare_one_user_with_additional(data, additional_data)  # (movies_count, )

        evaluated_movies = list(self.movies_le.transform(data[0])) + list(self.movies_le.transform(additional_data))

        return user_norm_ratings, user_mean, evaluated_movies

    @staticmethod
    def _extract_additional_data(kwargs):
        additional_data = []
        if "age" in kwargs:
            additional_data.append(Model_SVD_With_Features._age_to_age_group(kwargs["age"]))
        if "is_male" in kwargs:
            additional_data.append("gender_M" if kwargs["is_male"] else "gender_F")
        if "occupation" in kwargs:
            additional_data.append(f"occupation_{kwargs['occupation']}")
        return additional_data

    @staticmethod
    def _age_to_age_group(age: int):
        i = 1
        for i in range(1, len(config.available_age_groups)):
            if age < config.available_age_groups[i]:
                return config.available_age_groups[i - 1]
        return f"age_{config.available_age_groups[i - 1]}"

    def _prepare_one_user_with_additional(self, data: list, additional_data: list) -> (np.ndarray, float):
        user_ratings, user_mean = super(Model_SVD_With_Features, self)._prepare_one_user(data)
        user_ratings[self.movies_le.transform(additional_data)] = 1
        return user_ratings, user_mean

    def warmup(self, path: Optional[str | Path] = None) -> None:
        """
        Loads the model from ./data/model/. Refresh if it is already loaded.
        :param path: path for loading model. Default is config['model']
        """
        if path is None:
            path = config.model_features
        self._check_warmup_path(path)
        self.users_features = np.load(f"{path}/users_features.np.npy")
        self.movies_features = np.load(f"{path}/movies_features.np.npy")

        super(Model_SVD_With_Features, self).warmup(path)

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
        return super(Model_SVD_With_Features, self)._find_similar(movie_id, N, ignore_list=self.users_features)

    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save model weights to data/model
        :param path: path for saving model. Default is config['model']
        """
        if path is None:
            path = config.model_features
        logging.info(f"Save model into {path}...")
        path = Path(path)
        path.mkdir(exist_ok=True)
        super(Model_SVD_With_Features, self).save(path)
        np.save(str(path / "users_features.np"), self.users_features)
        np.save(str(path / "movies_features.np"), self.movies_features)
