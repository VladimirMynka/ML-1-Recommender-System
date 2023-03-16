from pathlib import Path
from typing import Optional


class Model:
    def __init__(self):
        pass

    def train(self, train_path: Path = None, **kwargs) -> None:
        """
        Receives the dataset filename. Performs model training. Saves the artifacts to ./model/. Logs the results.
        :param train_path: path to train dataset
        :param kwargs: another params for chosen model
        """
        pass

    def evaluate(self, data_path: Path, **kwargs) -> None:
        """
        Receives the dataset filename. Loads the model from ./data/model/.
        Evaluates the model with the provided dataset, prints the results and saves it to the log.
        :param data_path: path to evaluating data
        :param kwargs: another params for chosen model
        """
        pass

    def predict(self, data: list, top_m: int, **kwargs) -> list:
        """
        Get recommend movies for one user

        :param data: list `[[movie_id_1, movie_id_2, .., movie_id_N ], [rating_1, rating_2, .., rating_N]]` for one user
        :param top_m: how much recommended movies must be got
        :param kwargs: another params for chosen model

        :return: recommended movies with estimated rating in the same as input format
        """
        pass

    def warmup(self, path: Optional[str | Path] = None) -> None:
        """
        Loads the model from ./data/model/. Refresh if it is already loaded.
        :param path: path for loading model. Default is config['model']
        """
        pass

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
        pass

    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save model weights to data/model
        :param path: path for saving model. Default is config['model']
        """
        pass
