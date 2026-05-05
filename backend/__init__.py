from __future__ import annotations

from flask import Flask

from backend.config import Config
from backend.routes.about_routes import about_bp
from backend.routes.batch_routes import batch_bp
from backend.routes.explain_routes import explain_bp
from backend.routes.health_routes import health_bp
from backend.routes.molecule_routes import molecule_bp
from backend.routes.page_routes import page_bp
from backend.routes.predict_routes import predict_bp
from backend.utils.errors import register_error_handlers
from backend.utils.paths import ensure_runtime_directories


def create_app(config_object: type[Config] = Config) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(config_object.TEMPLATES_DIR),
        static_folder=str(config_object.STATIC_DIR),
    )
    app.config.from_object(config_object)
    ensure_runtime_directories()

    for blueprint in (
        page_bp,
        health_bp,
        about_bp,
        predict_bp,
        molecule_bp,
        explain_bp,
        batch_bp,
    ):
        app.register_blueprint(blueprint)

    register_error_handlers(app)
    return app

