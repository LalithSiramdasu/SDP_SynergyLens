from __future__ import annotations

from werkzeug.exceptions import BadRequest, HTTPException, InternalServerError, MethodNotAllowed, NotFound


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, code: str | None = None, status_code: int | None = None, details=None):
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details


class ArtifactError(AppError):
    status_code = 500
    code = "ARTIFACT_ERROR"


class ValidationError(AppError):
    status_code = 400
    code = "VALIDATION_ERROR"


class NotFoundError(AppError):
    status_code = 404
    code = "NOT_FOUND"


def register_error_handlers(app):
    from backend.utils.response import error_response

    @app.errorhandler(AppError)
    def handle_app_error(error: AppError):
        return error_response(error.code, error.message, error.status_code, error.details)

    @app.errorhandler(BadRequest)
    def handle_bad_request(error: BadRequest):
        return error_response("BAD_REQUEST", error.description or "Bad request.", 400)

    @app.errorhandler(NotFound)
    def handle_not_found(error: NotFound):
        return error_response("NOT_FOUND", "The requested resource was not found.", 404)

    @app.errorhandler(MethodNotAllowed)
    def handle_method_not_allowed(error: MethodNotAllowed):
        return error_response("METHOD_NOT_ALLOWED", "This HTTP method is not allowed for the route.", 405)

    @app.errorhandler(InternalServerError)
    def handle_internal_server_error(error: InternalServerError):
        original = getattr(error, "original_exception", None)
        if isinstance(original, AppError):
            return handle_app_error(original)
        app.logger.exception("Unhandled internal server error", exc_info=original or error)
        return error_response("INTERNAL_ERROR", "An internal backend error occurred.", 500)

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        if isinstance(error, HTTPException):
            return error_response(error.name.upper().replace(" ", "_"), error.description, error.code or 500)
        app.logger.exception("Unhandled backend error", exc_info=error)
        return error_response("INTERNAL_ERROR", "An internal backend error occurred.", 500)

