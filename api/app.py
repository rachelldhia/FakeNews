# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify
from flask_cors import CORS
# pyrefly: ignore [missing-import]
from flasgger import Swagger
# pyrefly: ignore [missing-import]
import joblib
import os
import logging
import sys

app = Flask(__name__)

# Mengizinkan CORS secara global dari semua domain (Vercel, Localhost, dll)
CORS(app, resources={r"/*": {"origins": "*"}})

# Inisialisasi Swagger untuk dokumentasi API interaktif
swagger = Swagger(app)

# Setup logging - tampilkan di stdout agar Railway bisa membaca logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info(f"=== FakeNews API Starting ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")

# Import sklearn - log versinya
try:
    import sklearn
    logger.info(f"scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    logger.error(f"scikit-learn tidak terinstall: {e}")

# ============================================================
# Fungsi untuk melatih model dari scratch jika tidak tersedia
# ============================================================
def train_fallback_model(save_path):
    """Train a simple fake news model with built-in sample data as fallback."""
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    logger.info("Melatih model fallback dengan data sampel...")

    # Sample training data (real=1, fake=0)
    train_texts = [
        # Real news samples (label=1)
        "The Federal Reserve raised interest rates by 25 basis points at its meeting Wednesday.",
        "Scientists published new research in Nature showing a link between sleep and memory.",
        "The Senate passed a bipartisan infrastructure bill with a vote of 69-30.",
        "WHO reports global measles cases declined by 18 percent over the past decade.",
        "Apple reported quarterly earnings of $94.8 billion, beating analyst expectations.",
        "NASA's James Webb Space Telescope captured images of galaxies 13 billion light-years away.",
        "The unemployment rate fell to 3.5 percent according to Bureau of Labor Statistics data.",
        "A federal judge ruled the regulation violated the Administrative Procedure Act.",
        "Researchers at Oxford University completed a clinical trial showing vaccine efficacy.",
        "The European Central Bank announced it would maintain current interest rate policy.",
        "Climate scientists recorded the hottest global average temperature since records began.",
        "The Supreme Court issued a ruling on the constitutionality of the voting rights act.",
        "New data from the Census Bureau shows population growth slowed in major cities.",
        "The company filed for Chapter 11 bankruptcy protection citing supply chain issues.",
        "Public health officials confirmed a cluster of cases linked to an outbreak investigation.",
        # Fake news samples (label=0)
        "BREAKING: Obama secretly signed a bill banning Christianity in America! Share before deleted!",
        "DOCTORS HATE HIM: This one weird trick cures diabetes in 48 hours with NO medicine!",
        "EXPOSED: The real reason vaccines cause autism that mainstream media won't report!!",
        "URGENT: Government microchips being secretly inserted in food supply chain - WAKE UP SHEEPLE",
        "Scientists ADMIT climate change is COMPLETELY FAKE and weather machines control storms!",
        "ALERT: Drinking bleach mixed with vinegar cures coronavirus in 24 hours!!",
        "Liberal Hollywood elites secretly fund underground baby harvesting operations!",
        "CONFIRMED: The moon landing was staged by NASA and Stanley Kubrick filmed it in Hollywood!",
        "THEY DON'T WANT YOU TO KNOW: Eating raw garlic daily cures all forms of cancer permanently!",
        "BREAKING: George Soros arrested for treason after secret recordings leaked to patriots!",
        "COVID vaccine makes you magnetic and connects your brain to 5G mind control network!!",
        "PROVEN: Drinking your own urine reverses aging and cures all autoimmune diseases!",
        "EXCLUSIVE: Deep state planning to cancel the 2024 elections using engineered pandemic!",
        "This retired nurse EXPOSES how pharmaceutical companies poison water supply for profit!!!",
        "Scientists SHOCKED: Earth is actually flat and NASA has been lying since 1969 SHARE NOW!",
    ]
    train_labels = [1]*15 + [0]*15

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])
    pipeline.fit(train_texts, train_labels)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Model fallback berhasil dilatih dan disimpan ke: {save_path}")
    return pipeline


# ============================================================
# Load atau train model
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"BASE_DIR: {BASE_DIR}")

# Coba semua kemungkinan path model
possible_paths = [
    os.path.join(BASE_DIR, '../models/fake_news_model.pkl'),
    os.path.join(BASE_DIR, 'models/fake_news_model.pkl'),
    os.path.join(BASE_DIR, 'fake_news_model.pkl'),
    os.path.join(os.getcwd(), 'models/fake_news_model.pkl'),
    os.path.join(os.getcwd(), 'fake_news_model.pkl'),
]

MODEL_PATH = None
for p in possible_paths:
    resolved = os.path.normpath(p)
    exists = os.path.exists(resolved)
    logger.info(f"  Cek path: {resolved} -> {'FOUND' if exists else 'not found'}")
    if exists and MODEL_PATH is None:
        MODEL_PATH = resolved

model_pipeline = None

if MODEL_PATH:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_pipeline = joblib.load(MODEL_PATH)
        logger.info(f"Model berhasil dimuat dari: {MODEL_PATH}")
        # Quick sanity test
        test_pred = model_pipeline.predict(["test news article"])
        logger.info(f"Model sanity check OK: predict={test_pred}")
    except Exception as e:
        logger.error(f"Gagal memuat model dari {MODEL_PATH}: {e}")
        model_pipeline = None

# Jika model tidak bisa dimuat, train fallback
if model_pipeline is None:
    try:
        fallback_path = os.path.join(BASE_DIR, 'fake_news_model_fallback.pkl')
        model_pipeline = train_fallback_model(fallback_path)
        logger.info("Menggunakan model fallback yang baru dilatih.")
    except Exception as e:
        logger.error(f"Gagal melatih model fallback: {e}")
        model_pipeline = None


# ============================================================
# Routes
# ============================================================

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
    import sklearn
    if model_pipeline:
        return jsonify({
            "status": "healthy",
            "message": "API and Model are ready!",
            "sklearn_version": sklearn.__version__,
            "model_type": str(type(model_pipeline).__name__)
        }), 200
    else:
        return jsonify({
            "status": "unhealthy",
            "message": "Model not loaded!",
            "sklearn_version": sklearn.__version__
        }), 503


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
        return jsonify({"error": "Model is not available on the server. Please check Railway logs."}), 500

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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model_pipeline.predict([text_input])[0]
            probabilities = model_pipeline.predict_proba([text_input])[0]

        # Menyusun response
        label = "Real News" if prediction == 1 else "Fake News"
        response = {
            "prediction": label,
            "confidence": {
                "fake_probability_percent": round(float(probabilities[0]) * 100, 2),
                "real_probability_percent": round(float(probabilities[1]) * 100, 2)
            }
        }
        logger.info(f"Prediksi: {label} | text length: {len(text_input)}")
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error saat prediksi: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Membaca PORT secara dinamis dari Environment Railway
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
