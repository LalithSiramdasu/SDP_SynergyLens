from __future__ import annotations

from flask import Blueprint, render_template


page_bp = Blueprint("pages", __name__)


@page_bp.get("/")
def index():
    return render_template("index.html")
