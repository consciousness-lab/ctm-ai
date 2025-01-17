from app_wrapper import FlaskAppWrapper
from flask import Flask


def create_app() -> Flask:
    """Factory function to create and configure the Flask application."""
    app_wrapper = FlaskAppWrapper()
    return app_wrapper.app
