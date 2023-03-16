import logging
from pathlib import Path
from typing import Optional

import fire

from src.my_models.model import Model
from src.my_models.model_svd import Model_SVD
from src.utils import init_logging

init_logging()


class My_Rec_Model:
    """
    Class-wrapper for Model to correct work in CLI
    """
    def __init__(self):
        self.model: Model = Model_SVD()

    def train(
        self,
        train_path: Path = None,
        save_path: Path = None,
        **kwargs
    ) -> None:
        """
        Receives the dataset filename. Performs model training. Saves the artifacts to ./model/. Logs the results
        :param train_path: path to train dataset
        :param save_path: path to save weights
        :param kwargs: another params for chosen model
        """
        self.model.train(train_path, **kwargs)
        self.save(save_path)

    def evaluate(
        self,
        data_path: Optional[Path | str] = None,
        load_path: Optional[Path | str] = None,
        **kwargs
    ) -> None:
        """
        Receives the dataset filename. Loads the model from ./data/model/.
        Evaluates the model with the provided dataset, prints the results and saves it to the log
        :param data_path: path to evaluating data
        :param load_path: path to load model
        :param kwargs: another params for chosen model
        """
        self.warmup(load_path)
        self.model.evaluate(data_path, **kwargs)

    def predict(
        self,
        data: list,
        top_m: int,
        load_path: Optional[Path | str] = None,
        **kwargs
    ) -> list:
        """
        Get recommend movies for one user

        :param data: list `[[movie_id_1, movie_id_2, .., movie_id_N ], [rating_1, rating_2, .., rating_N]]` for one user
        :param top_m: how much recommended movies must be got
        :param load_path: path to load model

        :return: recommended movies with estimated rating in the same as input format
        """
        self.warmup(load_path)
        return self.model.predict(data, top_m, **kwargs)

    def warmup(self, path: Optional[str | Path] = None) -> None:
        """
        Loads the model from ./data/model/. Refresh if it is already loaded
        :param path: path for loading model. Default is config['model']
        """
        self.model.warmup(path)

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
        return self.model.find_similar(movie_id, N)

    def save(self, path: Optional[str | Path] = None) -> None:
        """
        Save model weights to data/model
        :param path: path for saving model. Default is config['model']
        """
        self.model.save(path)


if __name__ == '__main__':
    fire.Fire(My_Rec_Model)
