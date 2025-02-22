from flask import Flask
from flask_cors import CORS
from app_wrapper import FlaskAppWrapper

app = FlaskAppWrapper()
CORS(app)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
