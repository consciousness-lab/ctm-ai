from app_wrapper import FlaskAppWrapper
from flask_cors import CORS

# Create an instance of your FlaskAppWrapper
flask_wrapper = FlaskAppWrapper()

# Enable CORS on the underlying Flask app - 允许所有来源（开发环境）
CORS(flask_wrapper.app, resources={r'/*': {'origins': '*'}})

# Export the underlying Flask app as the global WSGI callable.
app = flask_wrapper.app

if __name__ == '__main__':
    app.run(port=5000, debug=True)
