poetry install
poetry run gunicorn app:app --bind 0.0.0.0:5000
