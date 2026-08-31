"""Entry point: ``python app.py`` starts the development server.

For production use a WSGI server instead, e.g.::

    gunicorn "app:create_app()" --bind 0.0.0.0:8000
"""

from __future__ import annotations

import os

from src import create_app

app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    host = os.getenv("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=app.config["DEBUG"])
