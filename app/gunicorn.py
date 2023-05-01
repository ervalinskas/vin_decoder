import os

port = os.environ.get("PORT", 8000)
bind = f"0.0.0.0:{port}"
workers = os.getenv("GUNICORN_WORKER_NUM", 4)
timeout = 200
# user = 'root'
