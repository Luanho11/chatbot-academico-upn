"""
Rutas del sistema - Monitoreo de hardware y estado del sistema.
"""
from flask import Blueprint, jsonify

from hardware_sim import hardware
from services.ml_service import classifier, feedback_system
from services.groq_service import groq_service

system_bp = Blueprint('system', __name__)


@system_bp.route('/status')
def system_status():
    """Estado completo del sistema incluyendo hardware simulado."""
    status = hardware.get_full_status()
    return jsonify(status)


@system_bp.route('/hardware')
def hardware_status():
    """Estado de todos los sensores de hardware."""
    status = hardware.get_full_status()
    return jsonify(status['sensors'])


@system_bp.route('/hardware/<sensor>')
def sensor_status(sensor):
    """Estado de un sensor específico."""
    sensors = {
        'text_input': hardware.text_input,
        'nlp_processor': hardware.nlp_processor,
        'memory': hardware.memory,
        'network': hardware.network,
        'temperature': hardware.temperature,
        'power': hardware.power
    }
    
    if sensor not in sensors:
        return jsonify({'error': f'Sensor "{sensor}" no encontrado. Sensores disponibles: {list(sensors.keys())}'}), 404
    
    return jsonify(sensors[sensor].get_status())


@system_bp.route('/model')
def model_status():
    """Estado del modelo ML y Groq LLM."""
    metrics = classifier.get_metrics()
    feedback = feedback_system.get_stats()
    groq_stats = groq_service.get_stats()
    return jsonify({
        'model': metrics,
        'feedback': feedback,
        'groq': groq_stats
    })


@system_bp.route('/model/retrain', methods=['POST'])
def model_retrain():
    """Reentrena el modelo ML."""
    result = classifier.train()
    return jsonify(result)
