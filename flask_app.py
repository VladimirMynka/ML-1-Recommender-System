import logging

from flask import Flask, request, jsonify

from src.utils import init_logging, parse_credentials
from src.config import config
from src.services.model_service import model_service
from src.services.log_service import Log_Service

init_logging()
model_service = model_service()
log_service = Log_Service()

app = Flask(__name__)
app.logger = logging.getLogger()


@app.route('/api/predict', methods=["POST"])
def predict():
    args = request.json
    data = args.get("data")
    top_m = args.get("top_m")

    return jsonify(model_service.predict(data, top_m))


@app.route('/api/log', methods=["GET"])
def log():
    logs = log_service.get_log_rows()
    return jsonify({"logs": logs})


@app.route('/api/info', methods=["GET"])
def info():
    credentials = {}
    for key in config.credentials.storage:
        credentials.update(parse_credentials(config.credentials[key]))
    return jsonify(credentials)


@app.route('/api/reload', methods=["POST"])
def reload():
    model_service.model.warmup()
    return jsonify({"message": "Model reloaded"})


@app.route('/api/similar', methods=["POST"])
def similar():
    args = request.json
    data = args.get("movie_name")
    top_m = args.get("n")

    return jsonify(model_service.get_similar_by_name(data, top_m))


@app.errorhandler(500)
def some_error(e):
    logging.info(str(e))
    return str(e), 500


app.run("0.0.0.0", port=5017)
