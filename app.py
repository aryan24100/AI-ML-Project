import os

from flask import Flask, render_template, request, jsonify, session
from chatbot import get_response_with_state

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "iilm-chatbot-dev-secret")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("message", "")

    if not isinstance(user_input, str):
        return jsonify({"response": "Invalid request. Please send a text message."}), 400

    state = session.get("chat_state", {})
    response, new_state, suggestions = get_response_with_state(user_input, state)
    session["chat_state"] = new_state

    return jsonify({"response": response, "suggestions": suggestions})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)