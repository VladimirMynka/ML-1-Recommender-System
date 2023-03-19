### Without Docker:
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

### Flask App
For testing and debugging

```
nohup python -m flask_app &
```

### WSGI App \[[docs](https://flask.palletsprojects.com/en/2.2.x/deploying/gunicorn)\]
For production

- ```pip install gunicorn```
- ```gunicorn -w {n} 'wsgi_app:application'```, where n is count of processes will be started