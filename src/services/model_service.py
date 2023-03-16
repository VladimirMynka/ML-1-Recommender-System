import logging
from datetime import datetime

from src.my_models.model_svd import Model_SVD
from src.utils import save_credentials
from src.config import config

from fuzzywuzzy.fuzz import ratio


class model_service:
    def __init__(self):
        self.model = Model_SVD()
        try:
            self.model.warmup()
        except:
            logging.warning("There are not models in load path. Model will be trained")
            self.model.train()
            self.model.save()
            self.model.evaluate()
            save_credentials({
                'train_rmse': str(self.model.train_rmse),
                'test_rmse': str(self.model.test_rmse),
                'model_datetime': str(datetime.now())
            }, config.credentials.model)

    def get_movie_id_by_name(self, movie_name):
        similarities = self.model.df_movies.title.apply(lambda elem: ratio(movie_name, elem))
        movies_copy = self.model.df_movies.copy()
        movies_copy['sims'] = similarities
        movies_copy = movies_copy.sort_values('sims', ascending=False)
        return movies_copy.iloc[0]['movie_id']

    def predict(self, data, top_m):
        movie_names, ratings = data
        movie_ids = [self.get_movie_id_by_name(movie_name) for movie_name in movie_names]
        movie_new_ids, ratings = self.model.predict([movie_ids, ratings], top_m)
        movie_names = self.model.get_movies_names(movie_new_ids)
        return [movie_names, movie_ids]

    def get_similar_by_name(self, movie_name, n=20):
        movie_id = self.get_movie_id_by_name(movie_name)
        return self.model.find_similar(movie_id, n)
