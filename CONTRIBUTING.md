## Without Docker:
- Create directory `credentials` in the project root with file `credentials.txt` and fill information with fields:
  - Name
  - Surname
  - Group
- Create virtual environment with using python venv or conda [docs](https://docs.python.org/3/library/venv.html):
  - `python -m venv /path/to/new/virtual/environment`
- Activate virtual environment:
  - `source /path/to/new/virtual/environment/bin/activate`
- Install requirements:
  - `pip install -r requirements.txt`
- Get data:
  - `dvc pull`

### Flask App
For testing and debugging

```
nohup python -m flask_app &
```

Will run on port 5000. It can be changed at `src/config` file

### WSGI App \[[docs](https://flask.palletsprojects.com/en/2.2.x/deploying/gunicorn)\]
For production

- ```nohup gunicorn -w {n} 'wsgi_app:application' &```, where n is count of processes will be started

As default, it will run on port 8000. To change it, key `-b :{port}`, for example
```
nohup gunicorn -w {n} -b :5017 'wsgi_app:application' &
```