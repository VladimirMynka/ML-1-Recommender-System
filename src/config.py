from pathlib import Path

root_path = Path(__file__).parent.parent
dataset_path = root_path / 'data' / 'dataset'


class Config:
    def __init__(self, dictionary) -> None:
        self.storage = {}
        for key in dictionary:
            set_value = Config(dictionary[key]) if isinstance(dictionary[key], dict) else dictionary[key]
            self.__setattr__(key, set_value)
            self.storage[key] = set_value

    def __getitem__(self, item):
        return self.storage[item]

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Config(value)
        self.storage[key] = value
        self.__setattr__(key, value)


config = Config({
    'data': {
        'movies': {
            'path': dataset_path / 'movies.dat',
            'names': ['movie_id', 'title', 'genre'],
            'encoding': 'windows-1251'
        },
        'users': {
            'path': dataset_path / 'users.dat',
            'names': ['user_id', 'gender', 'age', 'occupation', 'zipcode']
        },
        'test': {
            'path': dataset_path / 'ratings_test.dat',
            'names': ['user_id', 'movie_id', 'rating', 'timestamp'],
        },
        'train': {
            'path': dataset_path / 'ratings_train.dat',
            'names': ['user_id', 'movie_id', 'rating', 'timestamp']
        }
    },
    'model': root_path / 'data' / 'model',
    'model_features': root_path / 'data' / 'model_upgraded',

    'logging': root_path / 'data' / 'log_file.log',
    'logging_rows_output': 20,
    'top_m_default': 5,
    'predict_type_default': 'by_movies_to_movies',

    'available_age_groups': [1, 18, 25, 35, 45, 50 ,56],

    'credentials': {
        'model': root_path / 'credentials' / 'model_credentials.txt',
        'model_features': root_path / 'credentials' / 'model_features_credentials.txt',
        'common': root_path / 'credentials' / 'credentials.txt',
        'docker': root_path / 'credentials' / 'docker_credentials.txt'
    },

    'flask_app_port': 5000
})
