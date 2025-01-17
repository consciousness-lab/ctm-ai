from app_wrapper import FlaskAppWrapper


def create_app():
    """Factory function to create and configure the Flask application."""
    app_wrapper = FlaskAppWrapper()
    return app_wrapper.app
