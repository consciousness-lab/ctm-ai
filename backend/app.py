from app_wrapper import FlaskAppWrapper

if __name__ == '__main__':
    app = FlaskAppWrapper()
    app.run(port=5000, debug=True)
