"""
Rutas del Dashboard - Panel de monitoreo del sistema.
"""
from flask import Blueprint, render_template

from hardware_sim import hardware
from services.ml_service import classifier, feedback_system, INTENT_DATA

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
def dashboard_page():
    """Página del dashboard de monitoreo."""
    hw = hardware.get_full_status()
    ml = classifier.get_metrics()
    fb = feedback_system.get_stats()
    
    intents = [{'id': k, 'examples': len(v['examples'])} for k, v in INTENT_DATA.items()]
    
    return render_template('dashboard.html',
                         hardware=hw,
                         model=ml,
                         feedback=fb,
                         intents=intents)
