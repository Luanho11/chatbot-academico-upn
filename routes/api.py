"""
Rutas API REST - Endpoints para integración con otros sistemas.
"""
import uuid
from flask import Blueprint, request, jsonify

from services.ml_service import classifier, response_generator, feedback_system, INTENT_DATA
from services.groq_service import groq_service
from hardware_sim import hardware

api_bp = Blueprint('api', __name__)


@api_bp.route('/v1/chat', methods=['POST'])
def api_chat():
    """
    API REST v1 - Endpoint principal de chat.
    
    Body:
        message (str): Mensaje del usuario
        user_id (str, opcional): ID del usuario
    
    Response:
        response: Respuesta del chatbot
        intent: Intención detectada
        confidence: Nivel de confianza
        hardware: Métricas de hardware simulado
    """
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({
            'error': 'Bad Request',
            'message': 'El campo "message" es requerido',
            'code': 400
        }), 400
    
    message = data['message'].strip()
    if not message:
        return jsonify({
            'error': 'Bad Request', 
            'message': 'El mensaje no puede estar vacío',
            'code': 400
        }), 400
    
    user_id = data.get('user_id', str(uuid.uuid4())[:8])
    
    # Procesar
    hw_status = hardware.process_chat_input(message)
    intent_result = classifier.predict(message)
    
    # Usar Groq LLM con fallback
    history = response_generator.get_history(user_id)
    groq_result = groq_service.chat(
        message=message,
        intent=intent_result['intent'],
        confidence=intent_result['confidence'],
        history=history
    )
    
    if groq_result['response']:
        bot_response = groq_result['response']
        source = 'groq'
    else:
        response_result = response_generator.generate_response(user_id, message, intent_result)
        bot_response = response_result['response']
        source = 'ml_fallback'
    
    return jsonify({
        'data': {
            'response': bot_response,
            'intent': intent_result['intent'],
            'confidence': intent_result['confidence'],
            'user_id': user_id,
            'request_id': hw_status['request_id'],
            'source': source
        },
        'meta': {
            'processing_time_ms': hw_status['nlp']['inference_time_ms'],
            'model_version': '1.0.0',
            'api_version': 'v1',
            'llm_model': groq_result.get('model', ''),
            'llm_tokens': groq_result.get('tokens', {})
        },
        'hardware': {
            'cpu_usage': hw_status['nlp']['gpu_usage'],
            'memory_mb': hw_status['memory']['ram_used'],
            'temperature_c': hw_status['temperature']['temperature'],
            'network_latency_ms': hw_status['network'].get('latency_ms', 0)
        }
    })


@api_bp.route('/v1/intents', methods=['GET'])
def api_intents():
    """Lista todas las intenciones disponibles."""
    intents = []
    for intent, data in INTENT_DATA.items():
        intents.append({
            'id': intent,
            'examples_count': len(data['examples']),
            'responses_count': len(data['responses'])
        })
    return jsonify({'data': intents, 'total': len(intents)})


@api_bp.route('/v1/intents/<intent>', methods=['GET'])
def api_intent_detail(intent):
    """Detalle de una intención específica."""
    if intent not in INTENT_DATA:
        return jsonify({'error': 'Not Found', 'message': f'Intención "{intent}" no encontrada'}), 404
    
    data = INTENT_DATA[intent]
    return jsonify({
        'data': {
            'id': intent,
            'examples': data['examples'],
            'responses_count': len(data['responses'])
        }
    })


@api_bp.route('/v1/predict', methods=['POST'])
def api_predict():
    """Predice la intención sin generar respuesta."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Bad Request', 'message': 'Campo "message" requerido'}), 400
    
    result = classifier.predict(data['message'])
    return jsonify({'data': result})


@api_bp.route('/v1/feedback', methods=['POST'])
def api_feedback():
    """Envía retroalimentación sobre una respuesta."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Bad Request', 'message': 'Datos requeridos'}), 400
    
    rating = data.get('rating', 0)
    if not (1 <= rating <= 5):
        return jsonify({'error': 'Bad Request', 'message': 'Rating debe ser entre 1 y 5'}), 400
    
    feedback = feedback_system.add_feedback(
        user_id=data.get('user_id', 'api_user'),
        message=data.get('message', ''),
        response=data.get('response', ''),
        rating=rating,
        comment=data.get('comment', '')
    )
    
    return jsonify({'data': feedback}), 201


@api_bp.route('/v1/stats', methods=['GET'])
def api_stats():
    """Estadísticas generales del sistema."""
    hw = hardware.get_full_status()
    ml = classifier.get_metrics()
    fb = feedback_system.get_stats()
    usage = response_generator.get_stats()
    
    return jsonify({
        'data': {
            'system': hw['system'],
            'model': ml,
            'feedback': fb,
            'usage': usage
        }
    })


@api_bp.route('/v1/hardware', methods=['GET'])
def api_hardware():
    """Estado del hardware simulado via API."""
    status = hardware.get_full_status()
    return jsonify({'data': status})


# Documentación de la API
@api_bp.route('/v1/docs', methods=['GET'])
def api_docs():
    """Documentación de la API REST."""
    return jsonify({
        'name': 'Chatbot Académico UPN - API REST',
        'version': '1.0.0',
        'description': 'API para el sistema inteligente de chatbot conversacional basado en NLP y Machine Learning',
        'endpoints': [
            {'method': 'POST', 'path': '/api/v1/chat', 'description': 'Enviar mensaje al chatbot'},
            {'method': 'GET', 'path': '/api/v1/intents', 'description': 'Listar intenciones disponibles'},
            {'method': 'GET', 'path': '/api/v1/intents/<intent>', 'description': 'Detalle de una intención'},
            {'method': 'POST', 'path': '/api/v1/predict', 'description': 'Predecir intención sin generar respuesta'},
            {'method': 'POST', 'path': '/api/v1/feedback', 'description': 'Enviar retroalimentación'},
            {'method': 'GET', 'path': '/api/v1/stats', 'description': 'Estadísticas generales'},
            {'method': 'GET', 'path': '/api/v1/hardware', 'description': 'Estado del hardware simulado'}
        ]
    })
