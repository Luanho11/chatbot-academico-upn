"""Utilidades varias."""
import uuid


def generate_user_id():
    """Genera un ID único de usuario."""
    return str(uuid.uuid4())[:8]
