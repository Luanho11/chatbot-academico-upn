# Sistema Inteligente de Chatbot Conversacional
## Basado en NLP y Machine Learning para la Asistencia Académica

**Autor:** Luis Angel Hostos Hostos  
**Universidad:** Universidad Privada del Norte

---

### Descripción

Chatbot conversacional que utiliza procesamiento de lenguaje natural (NLP) y Machine Learning para brindar asistencia académica 24/7 a estudiantes universitarios. Incluye simulación de hardware (sensores, procesador NLP, memoria, red, temperatura, alimentación) para monitoreo en tiempo real.

### Características

- **Chat inteligente** con clasificación de intenciones (9 categorías)
- **Modelo ML** simulado con TF-IDF y métricas de evaluación
- **Simulación de hardware** con 6 sensores virtuales
- **API REST** documentada para integración
- **Dashboard** de monitoreo en tiempo real
- **Sistema de feedback** para mejora continua

### Despliegue Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
python app.py

# Acceder
# http://localhost:5000
```

### Despliegue en Hosting Gratuito

#### Opción 1: Render.com
1. Crear cuenta en [render.com](https://render.com)
2. Conectar repositorio GitHub
3. Seleccionar "Web Service"
4. Configurar:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn wsgi:app`
5. Desplegar

#### Opción 2: PythonAnywhere
1. Crear cuenta en [pythonanywhere.com](https://pythonanywhere.com)
2. Subir archivos o clonar repositorio
3. Crear Web App con Flask
4. Configurar WSGI con `wsgi.py`

#### Opción 3: Railway
1. Crear cuenta en [railway.app](https://railway.app)
2. Conectar repositorio
3. Desplegar automáticamente

### Estructura del Proyecto

```
chatbot-academico/
├── app.py                    # Aplicación Flask principal
├── wsgi.py                   # Punto de entrada WSGI
├── requirements.txt          # Dependencias Python
├── Procfile                  # Para despliegue Heroku/Railway
├── render.yaml               # Para despliegue Render
├── hardware_sim/             # Simulación de hardware
│   ├── __init__.py
│   └── sensors.py            # Sensores simulados
├── services/                 # Servicios del chatbot
│   ├── __init__.py
│   └── ml_service.py         # Modelo ML y NLP
├── routes/                   # Rutas Flask
│   ├── chat.py               # Rutas del chat
│   ├── system.py             # Rutas del sistema
│   ├── api.py                # API REST
│   └── dashboard.py          # Dashboard
├── templates/                # Plantillas HTML
│   ├── index.html            # Página principal
│   ├── chat.html             # Interfaz de chat
│   └── dashboard.html        # Dashboard
├── static/                   # Archivos estáticos
│   ├── css/style.css
│   ├── js/chat.js
│   └── js/dashboard.js
└── utils/                    # Utilidades
    └── __init__.py
```

### API REST Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/chat/send` | Enviar mensaje al chatbot |
| POST | `/chat/feedback` | Enviar retroalimentación |
| GET | `/chat/history/<user_id>` | Historial de conversación |
| GET | `/system/status` | Estado completo del sistema |
| GET | `/system/hardware` | Estado de sensores |
| GET | `/system/model` | Métricas del modelo ML |
| POST | `/api/v1/chat` | API REST de chat |
| GET | `/api/v1/intents` | Listar intenciones |
| POST | `/api/v1/predict` | Predecir intención |
| POST | `/api/v1/feedback` | Retroalimentación |
| GET | `/api/v1/stats` | Estadísticas generales |
| GET | `/api/v1/hardware` | Estado hardware via API |
