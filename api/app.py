# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from flasgger import Swagger
# pyrefly: ignore [missing-import]
import joblib
import os
import logging

app = Flask(__name__)
CORS(app)
# Inisialisasi Swagger untuk dokumentasi API interaktif
swagger = Swagger(app)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Memuat model saat aplikasi Flask mulai berjalan
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/fake_news_model.pkl')
try:
    model_pipeline = joblib.load(MODEL_PATH)
    logging.info("Model berhasil dimuat ke dalam API.")
except Exception as e:
    logging.error(f"Gagal memuat model. Error: {e}")
    model_pipeline = None



@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint untuk mengecek status kesehatan API.
    ---
    responses:
      200:
        description: API dan Model berjalan dengan baik
      503:
        description: API berjalan tapi Model gagal dimuat
    """
    if model_pipeline:
        return jsonify({"status": "healthy", "message": "API and Model are ready!"}), 200
    else:
        return jsonify({"status": "unhealthy", "message": "Model not loaded!"}), 503


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint untuk memprediksi berita Fake atau Real.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              example: "Washington - The president signed a new bill today to improve infrastructure..."
    responses:
      200:
        description: Prediksi berhasil dilakukan
      400:
        description: Bad Request (Input JSON tidak valid atau kosong)
      500:
        description: Internal Server Error (Kendala pada server)
    """
    # Error Handling 1: Jika model belum siap
    if model_pipeline is None:
        return jsonify({"error": "Model is not available on the server."}), 500

    try:
        data = request.get_json(silent=True)

        # Error Handling 2: Jika user tidak mengirim format JSON yang benar
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid format. Please use JSON with a 'text' key."}), 400

        text_input = data['text']

        # Error Handling 3: Jika text kosong atau bukan string
        if not isinstance(text_input, str) or len(text_input.strip()) == 0:
            return jsonify({"error": "Text input cannot be empty and must be a string."}), 400

        # Melakukan prediksi
        prediction = model_pipeline.predict([text_input])[0]
        probabilities = model_pipeline.predict_proba([text_input])[0]

        # Menyusun response
        label = "Real News" if prediction == 1 else "Fake News"
        response = {
            "prediction": label,
            "confidence": {
                "fake_probability_percent": round(probabilities[0] * 100, 2),
                "real_probability_percent": round(probabilities[1] * 100, 2)
            }
        }
        return jsonify(response), 200

    except Exception as e:
        logging.error(f"Error saat prediksi: {e}")
        return jsonify({"error": "An internal server error occurred while processing the request."}), 500


if __name__ == '__main__':
    # Host 0.0.0.0 penting agar nanti bisa diakses dari dalam Docker container
    app.run(host='0.0.0.0', port=5001, debug=False)
