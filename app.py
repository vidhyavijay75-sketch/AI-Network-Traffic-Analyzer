from flask import Flask, request, jsonify, render_template
from predict import InvalidInputError, predict_congestion, load_model

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        load_model()
        return jsonify({
            "status": "ok",
            "model_loaded": True
        })
    except Exception as error:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "message": str(error)
        }), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "JSON input is required"
            }), 400

        result = predict_congestion(data)

        return jsonify(result)

    except InvalidInputError as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    print("Starting AI Network Traffic Analyzer...")
    app.run(host="127.0.0.1", port=5000, debug=False)