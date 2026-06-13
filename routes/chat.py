"""
Rutas del chat - Interfaz principal y endpoints de chat.
Usa Groq LLM como motor principal con fallback al modelo ML simulado.
"""
import uuid
from flask import Blueprint, render_template, request, jsonify, session

from services.ml_service import classifier, response_generator, feedback_system
from services.groq_service import groq_service
from hardware_sim import hardware

chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/chat')
def chat_page():
    """Página principal del chatbot."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())[:8]
    return render_template('chat.html', user_id=session.get('user_id'))


@chat_bp.route('/chat/send', methods=['POST'])
def chat_send():
    """Endpoint para enviar mensajes al chatbot."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Mensaje requerido'}), 400
    
    message = data['message'].strip()
    if not message:
        return jsonify({'error': 'El mensaje no puede estar vacío'}), 400
    
    user_id = data.get('user_id', session.get('user_id', str(uuid.uuid4())[:8]))
    
    # Procesar a través del simulador de hardware
    hw_status = hardware.process_chat_input(message)
    
    # Clasificar intención con modelo ML
    intent_result = classifier.predict(message)
    intent = intent_result['intent']
    confidence = intent_result['confidence']
    
    # Obtener historial reciente
    history = response_generator.get_history(user_id)
    
    # Intentar usar Groq LLM primero
    groq_result = groq_service.chat(
        message=message,
        intent=intent,
        confidence=confidence,
        history=history
    )
    
    if groq_result['response']:
        # Respuesta del LLM real (Groq)
        bot_response = groq_result['response']
        source = 'groq'
        llm_tokens = groq_result.get('tokens', {})
        llm_time = groq_result.get('response_time_s', 0)
    else:
        # Fallback al modelo ML simulado
        response_result = response_generator.generate_response(
            user_id, message, intent_result
        )
        bot_response = response_result['response']
        source = 'ml_fallback'
        llm_tokens = {}
        llm_time = 0
    
    # Guardar en historial del response_generator
    response_generator.conversation_history[user_id].append({
        'role': 'user',
        'text': message,
        'intent': intent,
        'confidence': confidence,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })
    response_generator.conversation_history[user_id].append({
        'role': 'assistant',
        'text': bot_response,
        'intent': intent,
        'source': source,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    })
    
    # Limitar historial
    if len(response_generator.conversation_history[user_id]) > 50:
        response_generator.conversation_history[user_id] = \
            response_generator.conversation_history[user_id][-30:]
    
    # Construir respuesta completa
    result = {
        'success': True,
        'response': bot_response,
        'intent': intent,
        'confidence': confidence,
        'probabilities': intent_result.get('probabilities', {}),
        'source': source,
        'hardware': {
            'processing_time_ms': hw_status['nlp']['inference_time_ms'],
            'gpu_usage': hw_status['nlp']['gpu_usage'],
            'memory_used_mb': hw_status['memory']['ram_used'],
            'temperature_c': hw_status['temperature']['temperature']
        },
        'llm': {
            'model': groq_result.get('model', ''),
            'tokens': llm_tokens,
            'response_time_s': llm_time,
            'source': source
        },
        'user_id': user_id,
        'request_id': hw_status['request_id']
    }
    
    return jsonify(result)


@chat_bp.route('/chat/feedback', methods=['POST'])
def chat_feedback():
    """Endpoint para enviar retroalimentación sobre una respuesta."""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Datos requeridos'}), 400
    
    user_id = data.get('user_id', 'anonymous')
    message = data.get('message', '')
    response = data.get('response', '')
    rating = data.get('rating', 0)
    comment = data.get('comment', '')
    
    if not (1 <= rating <= 5):
        return jsonify({'error': 'Rating debe ser entre 1 y 5'}), 400
    
    feedback = feedback_system.add_feedback(
        user_id, message, response, rating, comment
    )
    
    return jsonify({'success': True, 'feedback': feedback})


@chat_bp.route('/chat/history/<user_id>')
def chat_history(user_id):
    """Retorna el historial de conversación de un usuario."""
    history = response_generator.get_history(user_id)
    return jsonify({'user_id': user_id, 'history': history})
