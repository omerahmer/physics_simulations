import base64
from io import BytesIO

import matplotlib.pyplot as plt
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from simulation import (
    Chip,
)  # Assuming simulation.py is in the same directory and properly structured

app = Flask(__name__)
CORS(app)


@app.before_request
@app.route("/")
@app.route("/calculate_focal_length", methods=["POST", "GET"])
def calculate_focal_length():
    try:
        data = request.json  # doesn't work with decimal point
        spacings = list(map(float, data["spacings"]))  # doesn't work with decimal point
        thicknesses = list(
            map(float, data["thicknesses"])
        )  # doesn't work with decimal point
        diameters = float(data["diameter"])
        voltages = list(map(float, data["voltages"]))  # doesn't work with decimal point
        print(data)
        chip = Chip(spacings, thicknesses, diameters)
        focal_length = chip.get_system_focal_length(voltages)

        return jsonify({"focal_length": focal_length}), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 400


@app.route("/plot_ray", methods=["POST", "GET"])
def plot_ray():
    try:
        data = request.json
        angle = float(data["angle"])
        print(data["angle"])
        offset = float(data["offset"])
        energy = float(data["energy"])
        voltages = list(map(float, data["voltages"]))
        num_datapoints = int(data["num_datapoints"])

        spacings = list(map(float, data["spacings"]))
        thicknesses = list(map(float, data["thicknesses"]))
        diameter = float(data["diameter"])

        chip = Chip(spacings, thicknesses, diameter)
        buf = BytesIO()
        chip.plot_custom_ray(angle, offset, energy, voltages, num_datapoints)
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode("utf-8")
        return jsonify({"image": "data:image/png;base64," + image_base64}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=8000)
