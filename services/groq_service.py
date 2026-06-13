"""
Servicio de integración con Groq API (LLM real).
Utiliza la API de Groq para generar respuestas más inteligentes
usando modelos como Llama 3.3 70B mediante el SDK de OpenAI
(compatible con la API de Groq).
"""
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Configuración Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# System prompt para el chatbot académico
SYSTEM_PROMPT = """Eres un asistente académico inteligente de la Universidad Privada del Norte (UPN). Tu función es ayudar a estudiantes universitarios con sus consultas académicas.

Directrices:
- Responde siempre en español.
- Sé amable, paciente y académico en tus respuestas.
- Si te preguntan sobre un concepto, explícalo con claridad y pon un ejemplo.
- Si te piden ayuda con una tarea, guía paso a paso sin dar la respuesta completa directamente.
- Recomienda recursos educativos cuando sea apropiado (campus virtual, Coursera, Khan Academy, etc.).
- Si la pregunta no es académica, redirige amablemente al tema académico.
- No inventes información. Si no sabes algo, dilo con honestidad y sugiere dónde buscar.
- Mantén las respuestas concisas pero completas (2-4 párrafos máximo).
- Eres parte de un sistema que usa NLP y Machine Learning para entender la intención del estudiante.

Categorías de intención que manejas:
1. consulta_curso: Información sobre cursos y syllabus
2. duda_conceptual: Explicación de conceptos académicos
3. ayuda_tarea: Guía para resolver ejercicios y tareas
4. horario_consulta: Información sobre horarios de clases y tutorías
5. recurso_educativo: Recomendación de materiales de estudio
6. estado_academico: Consultas sobre notas y rendimiento
7. saludo: Saludos del estudiante
8. despedida: Despedidas y agradecimientos
9. soporte_tecnico: Problemas técnicos con plataformas universitarias
"""


class GroqService:
    """Servicio de integración con la API de Groq para generación de respuestas con LLM."""
    
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_MODEL
        self.available = True
        self.total_tokens_used = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.last_response_time = 0
        self._client = None
    
    def _get_client(self):
        """Obtiene el cliente de OpenAI configurado para Groq (lazy init)."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.groq.com/openai/v1"
                )
            except ImportError:
                logger.warning("OpenAI SDK no instalado. Usando fallback ML.")
                self.available = False
                return None
        return self._client
    
    def is_available(self):
        """Verifica si el servicio está disponible."""
        return self.api_key and self.available
    
    def chat(self, message, intent="", confidence=0.0, history=None):
        """
        Envía un mensaje al LLM de Groq y obtiene una respuesta.
        
        Args:
            message: Mensaje del usuario
            intent: Intención detectada por el clasificador ML
            confidence: Nivel de confianza del clasificador
            history: Historial de conversación reciente
        
        Returns:
            dict: Respuesta con texto, tokens y metadata
        """
        if not self.is_available():
            return {
                'response': None,
                'source': 'fallback',
                'error': 'Groq API no disponible'
            }
        
        client = self._get_client()
        if client is None:
            return {
                'response': None,
                'source': 'fallback',
                'error': 'Cliente OpenAI no inicializado'
            }
        
        # Construir mensajes
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Agregar contexto de intención
        intent_context = ""
        if intent:
            intent_labels = {
                'consulta_curso': 'Consulta sobre un curso universitario',
                'duda_conceptual': 'Duda sobre un concepto académico',
                'ayuda_tarea': 'Necesita ayuda con una tarea o ejercicio',
                'horario_consulta': 'Consulta sobre horarios de clase o tutorías',
                'recurso_educativo': 'Busca recursos o materiales de estudio',
                'estado_academico': 'Consulta sobre su estado académico o notas',
                'saludo': 'El estudiante está saludando',
                'despedida': 'El estudiante se despide',
                'soporte_tecnico': 'Problema técnico con plataformas universitarias'
            }
            intent_desc = intent_labels.get(intent, intent)
            intent_context = f"\n\n[Contexto del sistema ML - Intención detectada: {intent_desc} (confianza: {confidence:.1%})]"
        
        # Agregar historial reciente (últimos 6 mensajes)
        if history:
            for msg in history[-6:]:
                role = "user" if msg['role'] == 'user' else "assistant"
                messages.append({"role": role, "content": msg['text']})
        
        # Agregar mensaje actual con contexto
        user_content = message + intent_context
        messages.append({"role": "user", "content": user_content})
        
        # Llamar a la API de Groq via OpenAI SDK
        start_time = datetime.now()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            
            self.total_requests += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            self.last_response_time = round(elapsed, 3)
            
            text = response.choices[0].message.content
            tokens = {
                'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                'total_tokens': response.usage.total_tokens if response.usage else 0
            }
            
            self.total_tokens_used += tokens.get('total_tokens', 0)
            
            return {
                'response': text,
                'source': 'groq',
                'model': self.model,
                'tokens': tokens,
                'response_time_s': self.last_response_time,
                'intent_detected': intent,
                'confidence': confidence
            }
            
        except Exception as e:
            self.failed_requests += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            self.last_response_time = round(elapsed, 3)
            logger.error(f"Groq API error: {str(e)}")
            return {
                'response': None,
                'source': 'fallback',
                'error': str(e)
            }
    
    def get_stats(self):
        """Retorna estadísticas de uso de la API."""
        return {
            'available': self.available,
            'model': self.model,
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': round(
                (self.total_requests - self.failed_requests) / max(1, self.total_requests) * 100, 2
            ),
            'total_tokens_used': self.total_tokens_used,
            'last_response_time_s': self.last_response_time
        }


# Instancia global
groq_service = GroqService()
