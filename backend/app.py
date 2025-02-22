from app_wrapper import FlaskAppWrapper
from flask_cors import CORS

app = FlaskAppWrapper()
CORS(app.app, origins=['http://localhost:3000', 'http://18.224.61.142'])

if __name__ == '__main__':
    app.run(port=5000, debug=True)
