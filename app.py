"""
Sistema Inteligente de Chatbot Conversacional
Basado en Procesamiento de Lenguaje Natural y Machine Learning
para la Asistencia Académica mediante Servicios Web

Autor: Luis Angel Hostos Hostos
Universidad Privada del Norte
"""

import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'chatbot-academico-upn-2025')
    app.config['JSON_SORT_KEYS'] = False
    
    CORS(app)
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    # Registrar blueprints
    from routes.chat import chat_bp
    from routes.system import system_bp
    from routes.api import api_bp
    from routes.dashboard import dashboard_bp
    
    app.register_blueprint(chat_bp)
    app.register_blueprint(system_bp, url_prefix='/system')
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    
    # Ruta principal
    @app.route('/')
    def index():
        from flask import render_template
        return render_template('index.html')
    
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'Chatbot Académico UPN', 'version': '1.0.0'}
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
