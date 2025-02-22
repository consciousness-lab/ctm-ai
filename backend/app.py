from app_wrapper import FlaskAppWrapper
from flask_cors import CORS

# Create an instance of your FlaskAppWrapper
flask_wrapper = FlaskAppWrapper()

# Enable CORS on the underlying Flask app
CORS(flask_wrapper.app, origins=['http://localhost:3000', 'http://18.224.61.142'])

# Export the underlying Flask app as the global WSGI callable.
app = flask_wrapper.app

if __name__ == '__main__':
    app.run(port=5000, debug=True)

