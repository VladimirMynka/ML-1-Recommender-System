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

#### POST: `/api/predict`

Request example:

```json
{
  "predict_type": "by_movies_to_movies",
  "top_m": 5,
  "data": [
    [
      "movie_name_1",
      "movie_name_2",
      "movie_name_N"
    ],
    [
      5,
      5,
      4
    ]
  ]
}
``` 

Returns `top_m` (default 5) recommended movies with corresponding estimated rating. Sort descending. Response example:

```json
[
  [
    "movie_name_01",
    "movie_name_02",
    "movie_name_MM"
  ],
  [
    4.57,
    4.51,
    4.44
  ]
]
```

#### GET: `/api/log?n=20`

Returns last 20 rows of log. Response example:

```json
{
  "logs": "text"
}
```

#### GET: `/api/info`

Service Information: Credentials, Date and time of the build of the Docker image, Date, time and metrics of the training
of the currently deployed model.

Response example:

```json
{
  "DockerBuildDate": "Mon Jan 1 00:00:00 UTC 2000",
  "Group": "972001",
  "Name": "Name",
  "Surname": "Surname",
  "model_datetime": "2000-01-01 0:0:0.0",
  "test_rmse": "0.94",
  "train_rmse": "0.85"
}
```

#### POST: `/api/reload`

Reload the model. Response example:

```json
{
  "message": "text"
}
```

POST: `/api/similar`.

Request example:

```json
{
  "movie_name": "text",
  "n": 3
}
```

Returns list of `n` similar movies:

```json
[
  "movie_name_1",
  "movie_name_2",
  "movie_name_3"
]
```

## Docker

See [CONTRIBUTING.md](./CONTRIBUTING.md)