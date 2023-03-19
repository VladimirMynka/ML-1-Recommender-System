## Get data
- `pip install dvc`
- `dvc pull`

## CLI
Download repo by `git clone`

Then do
`pip install -r requirements.txt`

Then there are 4 available methods:

#### train
`python -m model train` - train model and save it
args:
- `--train_path` - path to train dataset. Default - from src.config
- `--save_path` - path to save model. Model will be saved as set of files
- `--d` - hyperparameter: count of hidden features
- `--kwargs` - if it is using not svd model

#### evaluate
`python -m model evaluate` - load model and evaluate it
args:
- `data_path` - path to validation dataset. Default is "data/test" from config
- `load_path` - path to load model weights. Default is "model" from config
- `--kwargs` - if it is using not svd model

#### predict
`python -m model predict` - load model and use it for predict movies for one user
args:
- `data` - list of movie_ids and their ratings
- `top_m` - how many movies must be found for this user
- `load_path` - path to load model weights. Default is "model" from config
- `--kwargs` - if it is using not svd model

#### find_similar
`python -m model find_similar` - load model and use it for search similar movies
args:
- `movie_id` - id of the movie for which another movies must found 
- `N` - how many movies must be found
- `load_path` - path to load model weights. Default is "model" from config


## API
api address: 2828.ftp.sh:5017

POST: `/api/predict`. Receives json with `data: [[movie_name_1, movie_name_2, .., movie_name_N ], [rating_1, rating_2, .., rating_N]]` and `top_m: int`, and returns `top_m` (default 5) recommended movies with corresponding estimated rating. Sort descending. `[[movie_name_1, movie_name_2, .., movie_name_M], [rating_1, rating_2, .., rating_M]]`

GET: `/api/log?n=20`. Last 20 rows of log.

GET: `/api/info`. Service Information: Credentials, Date and time of the build of the Docker image, Date, time and metrics of the training of the currently deployed model.

POST: `/api/reload`. Reload the model.

POST: `/api/similar`. Receives json with `"movie_name": string` and `"n": int`. Returns list of n similar movies also as json.

## Docker
See CONTRIBUTING.md