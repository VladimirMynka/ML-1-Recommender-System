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

