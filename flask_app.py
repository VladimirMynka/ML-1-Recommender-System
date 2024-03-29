import logging

from flask import Flask, request, jsonify

from src.config import config
from src.services.log_service import Log_Service
from src.services.model_service import Model_Service
from src.utils import init_logging, parse_credentials

init_logging()
model_service = Model_Service()
log_service = Log_Service()

app = Flask(__name__)
app.logger = logging.getLogger()


@app.route('/api/predict', methods=["POST"])
def predict():
    args = request.json
    data = args.get("data")
    top_m = args.get("top_m", default=5)
    predict_type = args.get("predict_type", default='by_movies_to_movies')

    return jsonify(model_service.predict(data, top_m, predict_type))


@app.route('/api/log', methods=["GET"])
def log():
    args = request.args
    lines = int(args.get("n", default=20))
    logs = log_service.get_log_rows(lines)
    return jsonify({"logs": logs})


@app.route('/api/info', methods=["GET"])
def info():
    credentials = {}
    for key in config.credentials.storage:
        credentials.update(parse_credentials(config.credentials[key]))
    return jsonify(credentials)


@app.route('/api/reload', methods=["POST"])
def reload():
    model_service.reload()
    return jsonify({"message": "Model reloaded"})


@app.route('/api/similar', methods=["POST"])
def similar():
    args = request.json
    data = args.get("movie_name")
    top_m = args.get("n")

    return jsonify(model_service.get_similar_by_name(data, top_m)[1])


@app.errorhandler(500)
def some_error(e):
    logging.info(str(e.original_exception))
    return f"Application Error: {e}", 500


if __name__ == "__main__":
    app.run("0.0.0.0", port=config.flask_app_port)
