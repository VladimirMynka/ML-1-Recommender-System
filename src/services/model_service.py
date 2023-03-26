import logging
from datetime import datetime

from fuzzywuzzy.fuzz import ratio

from src.config import config
from src.my_models.model_svd import Model_SVD
from src.my_models.model_svd_features import Model_SVD_With_Features
from src.utils import save_credentials, parse_credentials


class Model_Service:
    def __init__(self, model_type="with_features"):
        self.WITH_FEATURES = "with_features"

        if model_type == self.WITH_FEATURES:
            self.model = Model_SVD_With_Features()
        else:
            self.model = Model_SVD()

        self.credentials_file = config.credentials.model_features \
            if model_type == self.WITH_FEATURES \
            else config.credentials.model
        self.suffix = self.WITH_FEATURES if model_type == self.WITH_FEATURES else ""

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
            }, self.credentials_file, self.suffix)

    def get_movie_id_by_name(self, movie_name):
        similarities = self.model.df_movies.title.apply(lambda elem: ratio(movie_name, elem))
        movies_copy = self.model.df_movies.copy()
        movies_copy['sims'] = similarities
        movies_copy = movies_copy.sort_values('sims', ascending=False)
        return movies_copy.iloc[0]['movie_id']

    def predict(self, data, top_m, predict_type='by_movies_to_movies'):
        movie_names, ratings = data
        movie_ids = [self.get_movie_id_by_name(movie_name) for movie_name in movie_names]
        if predict_type == 'by_movies_to_movies':
            movie_new_ids, ratings = self.model.predict2([movie_ids, ratings], top_m)
        else:
            movie_new_ids, ratings = self.model.predict([movie_ids, ratings], top_m)
        movie_names = self.model.get_movies_names(movie_new_ids)
        return [movie_names.tolist(), ratings]

    def get_similar_by_name(self, movie_name, n=20):
        movie_id = self.get_movie_id_by_name(movie_name)
        return self.model.find_similar(movie_id, n)

    def reload(self):
        self.model.warmup()
        self.model.evaluate()
        credentials = parse_credentials(self.credentials_file)
        credentials['test_rmse'] = str(self.model.test_rmse)
        credentials['model_datetime'] = str(datetime.now())
        save_credentials(credentials, self.credentials_file, self.suffix)
