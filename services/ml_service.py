"""
Servicio de Machine Learning - Modelo de clasificación de intenciones
Simula el comportamiento de un modelo NLP+ML para asistencia académica.

Este módulo implementa:
- Preprocesamiento de texto (tokenización, normalización)
- Vectorización TF-IDF simulada
- Clasificación de intenciones con modelo simulado
- Generación de respuestas contextuales
- Sistema de retroalimentación (feedback)
"""

import re
import random
import math
import time
import json
from datetime import datetime
from collections import defaultdict


# Dataset de entrenamiento simulado (intenciones y ejemplos)
INTENT_DATA = {
    'consulta_curso': {
        'examples': [
            '¿de qué trata el curso de programación?',
            'qué temas se ven en cálculo',
            'cuál es el silabo de base de datos',
            'qué puedo aprender en inteligencia artificial',
            'cuáles son los temas de algebra lineal',
            'información sobre el curso de redes',
            'contenido del curso de estadística',
            'qué enseñan en ingeniería de software',
            'materias del curso de algoritmos',
            'temario de estructura de datos'
        ],
        'responses': [
            'El curso de {curso} abarca fundamentos teóricos y prácticos, incluyendo temas como introducción, conceptos clave, aplicaciones prácticas y proyectos integradores. Te recomiendo revisar el sílabo en el campus virtual para más detalles.',
            'En {curso} se cubren contenidos desde los conceptos básicos hasta aplicaciones avanzadas. El enfoque es práctico con proyectos que simulan situaciones reales del campo profesional.',
            'El programa de {curso} está diseñado para desarrollar competencias tanto técnicas como analíticas. Incluye evaluaciones continuas, trabajos prácticos y un proyecto final integrador.'
        ]
    },
    'duda_conceptual': {
        'examples': [
            'no entiendo qué es una función',
            'explícame qué es una variable',
            'qué significa herencia en programación',
            'cómo funciona un ciclo while',
            'qué es un algoritmo de ordenamiento',
            'no comprendo las derivadas',
            'qué es una base de datos relacional',
            'explícame el concepto de recursividad',
            'qué es POO',
            'cómo funciona una pila en estructuras de datos'
        ],
        'responses': [
            'Una {concepto} es un concepto fundamental que te permite {explicacion}. Piensa en ello como {analogia}. Te sugiero practicar con ejercicios del capítulo correspondiente para afianzar tu comprensión.',
            'El concepto de {concepto} puede parecer complejo al principio, pero es esencial para entender temas más avanzados. Básicamente, se trata de {explicacion}. Una forma sencilla de recordarlo es pensar que es como {analogia}.',
            'Déjame explicarte: {concepto} significa {explicacion}. Es similar a {analogia}. Te recomiendo revisar los ejemplos del material del curso y practicar con los ejercicios propuestos.'
        ]
    },
    'ayuda_tarea': {
        'examples': [
            'ayuda con mi tarea de programación',
            'cómo resuelvo este problema de matemáticas',
            'necesito ayuda con mi proyecto',
            'no sé cómo empezar mi trabajo',
            'guía para mi assignment',
            'cómo hago el ejercicio de bases de datos',
            'ayuda con laboratorio',
            'no puedo resolver el problema de algoritmos',
            'asistencia con mi práctica calificada',
            'necesito orientación para mi proyecto final'
        ],
        'responses': [
            'Para resolver tu tarea, te sugiero seguir estos pasos: 1) Lee cuidadosamente el enunciado completo, 2) Identifica los conceptos clave involucrados, 3) Desarrolla un plan de solución paso a paso, 4) Implementa y verifica cada parte. ¿En qué paso específico necesitas más ayuda?',
            'La mejor estrategia para abordar tu tarea es dividirla en subproblemas más pequeños. Comienza por entender qué se te pide, luego identifica las herramientas y conceptos necesarios. ¿Quieres que te guíe paso a paso?',
            'Te ayudo con gusto. Primero, asegúrate de entender los requisitos. Luego, identifica qué conceptos del curso aplican. Te recomiendo revisar ejemplos similares en el material de estudio. ¿Cuál es la parte que más te cuesta?'
        ]
    },
    'horario_consulta': {
        'examples': [
            'cuándo es la clase de programación',
            'a qué hora es la tutoría',
            'cuál es el horario de consultas',
            'cuándo puedo hablar con el profesor',
            'horario de asesorías',
            'qué días hay clase',
            'horario de atención del docente',
            'cuándo son las horas de oficina',
            'en qué horario se atienden consultas',
            'días de asesoría académica'
        ],
        'responses': [
            'Los horarios de consulta del docente suelen estar publicados en el campus virtual y en el sílabo del curso. Generalmente, las asesorías se brindan en horarios específicos cada semana. Te recomiendo verificar en la plataforma institucional para confirmar los horarios actualizados.',
            'Para conocer los horarios de consulta, puedes revisar el sílabo del curso o contactar directamente al docente. También puedes consultar en la secretaría de la escuela. Los horarios pueden variar cada ciclo académico.',
            'Los horarios de atención y tutorías están disponibles en el campus virtual. Si no los encuentras, te sugiero escribir al docente o revisar los anuncios del curso en la plataforma.'
        ]
    },
    'recurso_educativo': {
        'examples': [
            'dónde encuentro material de estudio',
            'recomiéndame libros de programación',
            'dónde hay tutoriales de python',
            'bibliografía recomendada',
            'recursos para aprender java',
            'dónde practicar ejercicios',
            'páginas para estudiar matemáticas',
            'material complementario',
            'videos educativos sobre algoritmos',
            'dónde descargo apuntes del curso'
        ],
        'responses': [
            'Te recomiendo estos recursos: 1) El campus virtual de la universidad tiene material oficial, 2) Plataformas como Coursera y edX ofrecen cursos gratuitos, 3) Documentación oficial del lenguaje o tecnología, 4) Canales de YouTube educativos como MIT OpenCourseWare. ¿Buscas algo en particular?',
            'Existen excelentes recursos para tu aprendizaje: libros de texto recomendados en el sílabo, plataformas en línea (Khan Academy, freeCodeCamp), y repositorios de ejercicios prácticos. La biblioteca virtual de la universidad también tiene acceso a ebooks y papers.',
            'Para profundizar en el tema, te sugiero: materiales del curso en el campus virtual, documentación técnica oficial, comunidades como Stack Overflow, y proyectos prácticos en GitHub. La combinación de teoría y práctica es clave.'
        ]
    },
    'estado_academico': {
        'examples': [
            'cuál es mi promedio',
            'cómo voy en el curso',
            'mis notas del parcial',
            'cuántos puntos necesito para aprobar',
            'cuál es mi situación académica',
            'verificar mi promedio ponderado',
            'calificación del proyecto',
            'notas del semestre',
            'cómo estoy en la materia',
            'progreso académico'
        ],
        'responses': [
            'Para consultar tu estado académico actual, debes ingresar al campus virtual con tus credenciales. Allí podrás ver tus notas, promedios y progreso en cada curso. Si tienes dudas sobre alguna evaluación, te sugiero contactar directamente al docente.',
            'Tu información académica está disponible en el sistema de gestión estudiantil. Puedes acceder desde el campus virtual para revisar calificaciones, asistencias y promedios. Si notas alguna discrepancia, comunícate con la coordinación académica.',
            'Para conocer tu rendimiento académico, revisa el portal del alumno en la plataforma institucional. Allí encontrarás tus calificaciones, porcentaje de asistencia y promedio acumulado. Recuerda que también puedes solicitar constancias oficiales.'
        ]
    },
    'saludo': {
        'examples': [
            'hola',
            'buenos días',
            'buenas tardes',
            'hey',
            'qué tal',
            'buenas noches',
            'hola chatbot',
            'saludos',
            'qué onda',
            'buenas'
        ],
        'responses': [
            '¡Hola! Soy el asistente académico de la universidad. Estoy aquí para ayudarte con tus consultas sobre cursos, conceptos, tareas, horarios y más. ¿En qué puedo asistirte hoy?',
            '¡Buenos días! Bienvenido al sistema de asistencia académica. Puedo ayudarte con dudas sobre tus cursos, recursos de estudio, horarios y más. ¿Qué necesitas saber?',
            '¡Saludos! Soy tu asistente académico virtual. Estoy disponible 24/7 para apoyarte en tu aprendizaje. ¿Tienes alguna consulta o duda que pueda resolver?'
        ]
    },
    'despedida': {
        'examples': [
            'gracias',
            'adiós',
            'hasta luego',
            'chao',
            'nos vemos',
            'me voy',
            'hasta la próxima',
            'bye',
            'muchas gracias por la ayuda',
            'te lo agradezco'
        ],
        'responses': [
            '¡De nada! Recuerda que estoy disponible las 24 horas para ayudarte. ¡Mucho éxito en tus estudios! No dudes en consultarme cuando lo necesites.',
            '¡Hasta pronto! Estaré aquí cuando me necesites. Sigue estudiando y no dudes en volver si tienes más dudas. ¡Tú puedes!',
            '¡Con gusto! Me alegra haber podido ayudarte. Recuerda que puedes consultarme en cualquier momento. ¡Éxito en tu aprendizaje!'
        ]
    },
    'soporte_tecnico': {
        'examples': [
            'no funciona el campus virtual',
            'error al subir mi tarea',
            'problema con mi cuenta',
            'no puedo acceder al sistema',
            'el enlace no funciona',
            'cómo cambio mi contraseña',
            'falla en la plataforma',
            'no carga la página',
            'problema técnico',
            'ayuda con el sistema'
        ],
        'responses': [
            'Para problemas técnicos, te recomiendo: 1) Intentar limpiar caché y cookies del navegador, 2) Probar con otro navegador, 3) Verificar tu conexión a internet. Si el problema persiste, contacta al soporte técnico de la universidad o reporta el incidente a través del portal de ayuda.',
            'Los inconvenientes técnicos suelen resolverse con pasos simples: actualiza tu navegador, verifica tu conexión, o intenta desde otro dispositivo. Si el problema continúa, el equipo de soporte de TI de la universidad puede ayudarte. Puedes reportar el issue en el portal de soporte.',
            'Lamento el inconveniente. Primero, intenta cerrar sesión y volver a ingresar. Si eso no funciona, prueba desde una ventana de incógnito. Para problemas persistentes, contacta al centro de soporte técnico de la universidad con los detalles del error.'
        ]
    }
}

# Conceptos y analogías para respuestas dinámicas
CONCEPT_EXPLANATIONS = {
    'función': {
        'explicacion': 'asociar una entrada con una salida específica mediante un conjunto de instrucciones',
        'analogia': 'una máquina que recibe un ingrediente y produce un resultado específico'
    },
    'variable': {
        'explicacion': 'almacenar un valor que puede cambiar durante la ejecución del programa',
        'analogia': 'una caja etiquetada donde puedes guardar y cambiar cosas'
    },
    'herencia': {
        'explicacion': 'reutilizar atributos y métodos de una clase padre en una clase hija',
        'analogia': 'un hijo que hereda características de sus padres pero también tiene las suyas propias'
    },
    'ciclo while': {
        'explicacion': 'repetir un bloque de código mientras se cumpla una condición',
        'analogia': 'seguir corriendo mientras no estés cansado'
    },
    'algoritmo': {
        'explicacion': 'un conjunto finito de pasos ordenados para resolver un problema',
        'analogia': 'una receta de cocina paso a paso'
    },
    'derivada': {
        'explicacion': 'medir la tasa de cambio instantánea de una función en un punto',
        'analogia': 'el velocímetro de un auto que te dice qué tan rápido vas en un instante'
    },
    'base de datos relacional': {
        'explicacion': 'organizar la información en tablas conectadas mediante relaciones',
        'analogia': 'un archivero donde cada gaveta está conectada con otras mediante etiquetas'
    },
    'recursividad': {
        'explicacion': 'una función que se llama a sí misma para resolver subproblemas más pequeños',
        'analogia': 'las muñecas rusas que contienen otra más pequeña dentro'
    },
    'POO': {
        'explicacion': 'organizar el código en objetos que combinan datos y comportamiento',
        'analogia': 'construir con bloques LEGO donde cada pieza tiene forma y función específica'
    },
    'pila': {
        'explicacion': 'una estructura de datos donde el último en entrar es el primero en salir (LIFO)',
        'analogia': 'una pila de platos donde solo puedes tomar el de arriba'
    }
}

CURSO_NAMES = ['Programación', 'Cálculo', 'Base de Datos', 'Inteligencia Artificial',
               'Álgebra Lineal', 'Redes', 'Estadística', 'Ingeniería de Software',
               'Algoritmos', 'Estructura de Datos']


class TextPreprocessor:
    """Preprocesador de texto para NLP."""
    
    def __init__(self):
        self.stop_words = {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
                          'de', 'del', 'al', 'a', 'en', 'por', 'para', 'con', 'sin',
                          'sobre', 'entre', 'y', 'o', 'pero', 'si', 'no', 'que', 'qué',
                          'es', 'son', 'se', 'su', 'mi', 'tu', 'muy', 'más', 'menos',
                          'este', 'esta', 'eso', 'esa', 'estos', 'estas', 'esos', 'esas'}
    
    def normalize(self, text):
        """Normaliza el texto: minúsculas, sin acentos, sin caracteres especiales."""
        text = text.lower().strip()
        # Reemplazar acentos
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ü': 'u'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        # Remover caracteres especiales excepto espacios y ?!
        text = re.sub(r'[^a-z0-9\s?!]', '', text)
        return text
    
    def tokenize(self, text):
        """Tokeniza el texto en palabras."""
        normalized = self.normalize(text)
        tokens = normalized.split()
        return tokens
    
    def remove_stop_words(self, tokens):
        """Remueve stop words de los tokens."""
        return [t for t in tokens if t not in self.stop_words]
    
    def preprocess(self, text):
        """Pipeline completo de preprocesamiento."""
        tokens = self.tokenize(text)
        tokens = self.remove_stop_words(tokens)
        return tokens


class SimulatedTFIDF:
    """Vectorizador TF-IDF simulado."""
    
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.idf = {}
    
    def fit(self, documents):
        """Simula el ajuste del vectorizador TF-IDF."""
        # Generar vocabulario simulado basado en las palabras de los documentos
        all_words = set()
        for doc in documents:
            words = doc.lower().split()
            all_words.update(words)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(all_words)[:self.vocab_size])}
        
        # Simular valores IDF
        n_docs = len(documents)
        for word in self.vocabulary:
            # IDF simulado: log(N / df) donde df es aleatorio
            df = random.randint(1, max(2, n_docs // 2))
            self.idf[word] = math.log(n_docs / df) + 1
    
    def transform(self, text):
        """Transforma texto en vector TF-IDF simulado."""
        vector = [0.0] * self.vocab_size
        words = text.lower().split()
        
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                # TF simulado
                tf = words.count(word) / len(words) if words else 0
                # TF-IDF
                idf = self.idf.get(word, 1.0)
                vector[idx] = tf * idf
        
        # Añadir ruido simulado para realismo
        for i in range(self.vocab_size):
            if vector[i] == 0:
                vector[i] = random.uniform(0, 0.01)
        
        # Normalizar
        norm = math.sqrt(sum(v**2 for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector


class SimulatedIntentClassifier:
    """Clasificador de intenciones simulado basado en similitud."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = SimulatedTFIDF()
        self.trained = False
        self.training_data = []
        self.training_time = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1_score = 0
    
    def train(self):
        """Entrena el modelo con los datos de intenciones."""
        start_time = time.time()
        
        # Preparar datos de entrenamiento
        all_examples = []
        self.training_data = []
        
        for intent, data in INTENT_DATA.items():
            for example in data['examples']:
                processed = self.preprocessor.preprocess(example)
                all_examples.append(' '.join(processed))
                self.training_data.append({
                    'text': example,
                    'processed': processed,
                    'intent': intent
                })
        
        # Ajustar vectorizador
        self.vectorizer.fit(all_examples)
        
        # Simular métricas de entrenamiento
        self.training_time = time.time() - start_time
        self.accuracy = round(random.uniform(0.88, 0.96), 4)
        self.precision = round(random.uniform(0.86, 0.95), 4)
        self.recall = round(random.uniform(0.87, 0.94), 4)
        self.f1_score = round(2 * (self.precision * self.recall) / 
                             (self.precision + self.recall), 4)
        
        self.trained = True
        return {
            'status': 'trained',
            'training_time_s': round(self.training_time, 3),
            'samples': len(self.training_data),
            'intents': len(INTENT_DATA),
            'metrics': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score
            }
        }
    
    def predict(self, text):
        """Predice la intención del texto."""
        if not self.trained:
            self.train()
        
        processed = self.preprocessor.preprocess(text)
        input_text = ' '.join(processed)
        
        # Calcular similitud con cada ejemplo de entrenamiento
        best_intent = 'saludo'
        best_score = 0
        scores = {}
        
        for intent in INTENT_DATA:
            intent_score = self._calculate_similarity(input_text, intent)
            scores[intent] = round(intent_score, 4)
            if intent_score > best_score:
                best_score = intent_score
                best_intent = intent
        
        # Umbral de confianza
        confidence = best_score
        if confidence < 0.3:
            best_intent = 'saludo'  # Default a saludo si no hay confianza
        
        # Simular probabilidad softmax
        total = sum(math.exp(s * 3) for s in scores.values())
        probabilities = {k: round(math.exp(v * 3) / total, 4) for k, v in scores.items()}
        
        return {
            'intent': best_intent,
            'confidence': round(confidence, 4),
            'probabilities': probabilities,
            'processed_text': input_text,
            'tokens': processed
        }
    
    def _calculate_similarity(self, input_text, intent):
        """Calcula similitud simulada entre texto y una intención."""
        input_words = set(input_text.split())
        intent_words = set()
        
        for example in INTENT_DATA[intent]['examples']:
            words = self.preprocessor.preprocess(example)
            intent_words.update(words)
        
        if not input_words or not intent_words:
            return 0
        
        # Similitud Jaccard con boost por coincidencias clave
        intersection = input_words.intersection(intent_words)
        union = input_words.union(intent_words)
        
        jaccard = len(intersection) / len(union) if union else 0
        
        # Boost por coincidencias de palabras clave
        key_word_matches = len(intersection)
        boost = min(0.5, key_word_matches * 0.12)
        
        return min(1.0, jaccard + boost)
    
    def get_metrics(self):
        """Retorna las métricas del modelo."""
        return {
            'trained': self.trained,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'training_samples': len(self.training_data),
            'num_intents': len(INTENT_DATA),
            'training_time_s': round(self.training_time, 3)
        }


class ResponseGenerator:
    """Generador de respuestas contextuales."""
    
    def __init__(self, classifier):
        self.classifier = classifier
        self.preprocessor = TextPreprocessor()
        self.conversation_history = defaultdict(list)
        self.session_context = defaultdict(dict)
    
    def generate_response(self, user_id, text, intent_result):
        """Genera una respuesta contextual basada en la intención detectada."""
        intent = intent_result['intent']
        confidence = intent_result['confidence']
        
        # Obtener plantillas de respuesta
        responses = INTENT_DATA[intent]['responses']
        template = random.choice(responses)
        
        # Personalizar respuesta según la intención
        response = self._customize_response(template, text, intent)
        
        # Añadir contexto de conversación
        context_info = self._build_context_info(user_id, intent)
        if context_info:
            response += '\n\n' + context_info
        
        # Guardar en historial
        self.conversation_history[user_id].append({
            'role': 'user',
            'text': text,
            'intent': intent,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        self.conversation_history[user_id].append({
            'role': 'assistant',
            'text': response,
            'intent': intent,
            'timestamp': datetime.now().isoformat()
        })
        
        # Limitar historial
        if len(self.conversation_history[user_id]) > 50:
            self.conversation_history[user_id] = self.conversation_history[user_id][-30:]
        
        return {
            'response': response,
            'intent': intent,
            'confidence': confidence,
            'probabilities': intent_result.get('probabilities', {}),
            'user_id': user_id
        }
    
    def _customize_response(self, template, text, intent):
        """Personaliza la plantilla de respuesta con información del texto."""
        # Para duda conceptual, extraer concepto
        if intent == 'duda_conceptual':
            concept = self._extract_concept(text)
            if concept in CONCEPT_EXPLANATIONS:
                explanation = CONCEPT_EXPLANATIONS[concept]
                return template.format(
                    concepto=concept,
                    explicacion=explanation['explicacion'],
                    analogia=explanation['analogia']
                )
            else:
                return template.format(
                    concepto=concept,
                    explicacion='un elemento clave del curso que requiere práctica para dominar',
                    analogia='una pieza de un rompecabezas que encaja con otras para formar el conocimiento completo'
                )
        
        # Para consulta de curso, extraer nombre del curso
        if intent == 'consulta_curso':
            curso = self._extract_curso(text)
            return template.format(curso=curso)
        
        return template
    
    def _extract_concept(self, text):
        """Extrae el concepto principal del texto."""
        text_lower = text.lower()
        for concept in CONCEPT_EXPLANATIONS:
            if concept.lower() in text_lower:
                return concept
        # Extraer palabras clave después de "qué es" o "explícame"
        for pattern in [r'qué es (?:una? )?(.+?)[\?\.]', r'explícame (?:qué es )?(.+?)[\?\.]',
                       r'no entiendo (?:qué es )?(.+?)[\?\.]', r'cómo funciona (?:una? )?(.+?)[\?\.]']:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip()
        return 'este concepto'
    
    def _extract_curso(self, text):
        """Extrae el nombre del curso del texto."""
        text_lower = text.lower()
        for curso in CURSO_NAMES:
            if curso.lower() in text_lower:
                return curso
        # Buscar después de "curso de"
        match = re.search(r'curso de (.+?)[\?\.]', text_lower)
        if match:
            return match.group(1).strip().title()
        return 'este curso'
    
    def _build_context_info(self, user_id, current_intent):
        """Construye información contextual basada en el historial."""
        history = self.conversation_history.get(user_id, [])
        if len(history) < 2:
            return ''
        
        # Contar intenciones previas
        intent_counts = defaultdict(int)
        for msg in history:
            if msg['role'] == 'user':
                intent_counts[msg['intent']] += 1
        
        recent_intents = [msg['intent'] for msg in history[-6:] if msg['role'] == 'user']
        
        context_parts = []
        if intent_counts['duda_conceptual'] >= 3:
            context_parts.append('📌 Veo que tienes varias dudas conceptuales. Te recomiendo repasar los fundamentos del curso.')
        if intent_counts['ayuda_tarea'] >= 2:
            context_parts.append('📚 Recuerda que es importante intentar resolver los ejercicios por tu cuenta antes de buscar ayuda.')
        if 'duda_conceptual' in recent_intents and current_intent == 'ayuda_tarea':
            context_parts.append('💡 Parece que las dudas conceptuales están afectando tus tareas. Repasar la teoría puede ayudarte.')
        
        return '\n'.join(context_parts)
    
    def get_history(self, user_id):
        """Retorna el historial de conversación."""
        return self.conversation_history.get(user_id, [])
    
    def get_stats(self, user_id=None):
        """Retorna estadísticas de uso."""
        if user_id:
            history = self.conversation_history.get(user_id, [])
            return {
                'user_id': user_id,
                'total_messages': len(history),
                'intent_distribution': dict(
                    defaultdict(int, 
                        {msg['intent']: 1 for msg in history if msg['role'] == 'user'})
                )
            }
        
        total_users = len(self.conversation_history)
        total_messages = sum(len(h) for h in self.conversation_history.values())
        intent_dist = defaultdict(int)
        for history in self.conversation_history.values():
            for msg in history:
                if msg['role'] == 'user':
                    intent_dist[msg['intent']] += 1
        
        return {
            'total_users': total_users,
            'total_messages': total_messages,
            'intent_distribution': dict(intent_dist)
        }


class FeedbackSystem:
    """Sistema de retroalimentación para mejorar el modelo."""
    
    def __init__(self):
        self.feedback_data = []
    
    def add_feedback(self, user_id, message, response, rating, comment=''):
        """Añade retroalimentación del usuario."""
        feedback = {
            'user_id': user_id,
            'message': message,
            'response': response,
            'rating': rating,  # 1-5
            'comment': comment,
            'timestamp': datetime.now().isoformat()
        }
        self.feedback_data.append(feedback)
        return feedback
    
    def get_stats(self):
        """Retorna estadísticas de retroalimentación."""
        if not self.feedback_data:
            return {'total_feedback': 0, 'avg_rating': 0}
        
        ratings = [f['rating'] for f in self.feedback_data]
        return {
            'total_feedback': len(self.feedback_data),
            'avg_rating': round(sum(ratings) / len(ratings), 2),
            'rating_distribution': {
                str(i): ratings.count(i) for i in range(1, 6)
            },
            'recent_feedback': self.feedback_data[-5:]
        }


# Instancias globales
classifier = SimulatedIntentClassifier()
response_generator = ResponseGenerator(classifier)
feedback_system = FeedbackSystem()

# Entrenar modelo al importar
classifier.train()
