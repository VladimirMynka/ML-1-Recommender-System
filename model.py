from pathlib import Path

class My_Rec_Model:
    def __init__(self) -> None:
        pass

    def train(self, train_path: Path) -> None:
        """
        Recieves the dataset filename. Performes model training. Saves the artifacts to ./model/. Logs the results.

        :param train_path: path to train dataset
        """
        pass

    def evaluate(self, data_path: Path) -> None:
        """
        Recieves the dataset filename. Loads the model from ./data/model/. Evaluates the model with the provided dataset, prints the results and saves it to the log.

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
        pass

    def warmup(self) -> None:
        """
        Loads the model from ./data/model/. Refresh if it is already loaded.
        """
        pass

    def find_similar(self, movie_id, N=5) -> list:
        """
        Returns N (parameter, default=5) most similar movies for input movie_id

        :param movie_id: id of a movie for which must be found similar some

        :return: Returns list [[movie_id_1, movie_id_2, .., movie_id_N ], [[movie_name_1, movie_name_2, .., movie_name_N]]] Descending sorting by similarity
        """
        pass


if __name__ == '__main__':
    pass
