import os
from io import BytesIO

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

MODEL_NAME = "gemma-4-26b-a4b-it"


@app.route("/")
def home():
    return render_template("index.html")


@app.errorhandler(413)
def file_too_large(error):
    return jsonify(
        error="Image is too large. Maximum size is 5 MB."
    ), 413


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("screenshot")
    problem = request.form.get("problem", "").strip()

    if not uploaded or not uploaded.filename:
        return jsonify(error="Please upload a screenshot."), 400

    image_bytes = uploaded.read()

    if not image_bytes:
        return jsonify(error="The uploaded image is empty."), 400

    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()

        if image.format not in ("PNG", "JPEG", "WEBP"):
            return jsonify(
                error="Please use PNG, JPG or WEBP."
            ), 400

        mime_type = {
            "PNG": "image/png",
            "JPEG": "image/jpeg",
            "WEBP": "image/webp",
        }[image.format]

    except (UnidentifiedImageError, OSError):
        return jsonify(error="Could not read this image."), 400

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return jsonify(
            error="API key missing. Configure GEMINI_API_KEY in .env."
        ), 500

    prompt = f"""
You are DJJO AI, a helpful visual troubleshooting assistant.

Analyze the supplied screenshot and help the user understand it.

User's description: {problem or "Not provided"}

Use these sections:

WHAT I SEE
Describe the visible screen.

POSSIBLE ISSUE
Explain the likely issue and mention uncertainty.

STEP-BY-STEP GUIDANCE
Give numbered steps grounded in what is visible.
Do not invent buttons or claim to have clicked anything.

IMPORTANT
Protect passwords and private information.
Ask a clarifying question if the screenshot is insufficient.
"""

    try:
        client = genai.Client(api_key=api_key)

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[image_part, prompt],
        )

        if not response.text:
            return jsonify(error="The AI returned an empty answer."), 502

        return jsonify(answer=response.text)

    except Exception:
        app.logger.exception("DJJO AI request failed")
        return jsonify(
            error=(
                "AI analysis failed. Check your API key, "
                "model availability, internet connection and quota."
            )
        ), 502


if __name__ == "__main__":
    app.run(debug=True)