/**
 * Chatbot Académico UPN - Chat JavaScript
 * Maneja la interacción del chat, envío de mensajes y métricas en vivo.
 */

let lastBotResponse = '';
let lastUserMessage = '';

// Enviar mensaje
function sendMessage() {
    const input = document.getElementById('messageInput');
    const message = input.value.trim();
    if (!message) return;
    
    input.value = '';
    lastUserMessage = message;
    
    // Ocultar sugerencias
    const suggestions = document.getElementById('quickSuggestions');
    if (suggestions) suggestions.style.display = 'none';
    
    // Agregar mensaje del usuario
    addMessage('user', message);
    
    // Mostrar indicador de escritura
    showTyping();
    
    // Enviar al servidor
    fetch('/chat/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: message,
            user_id: document.getElementById('userId').textContent
        })
    })
    .then(res => res.json())
    .then(data => {
        removeTyping();
        if (data.success) {
            lastBotResponse = data.response;
            addMessage('bot', data.response, data.intent, data.confidence, data.source);
            updateMetrics(data);
            showFeedback();
        } else {
            addMessage('bot', 'Lo siento, ocurrió un error al procesar tu mensaje. Intenta de nuevo.');
        }
    })
    .catch(err => {
        removeTyping();
        addMessage('bot', 'Error de conexión. Por favor, verifica tu conexión a internet e intenta nuevamente.');
        console.error('Error:', err);
    });
}

// Enviar mensaje rápido
function sendQuickMessage(text) {
    document.getElementById('messageInput').value = text;
    sendMessage();
}

// Agregar mensaje al chat
function addMessage(role, text, intent, confidence, source) {
    const container = document.getElementById('chatMessages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    
    const now = new Date();
    const timeStr = now.getHours().toString().padStart(2, '0') + ':' + 
                    now.getMinutes().toString().padStart(2, '0');
    
    let avatarIcon = role === 'bot' ? '<i class="bi bi-robot"></i>' : '<i class="bi bi-person-fill"></i>';
    let intentBadge = '';
    if (intent && role === 'bot') {
        const intentLabels = {
            'consulta_curso': 'Consulta de Curso',
            'duda_conceptual': 'Duda Conceptual',
            'ayuda_tarea': 'Ayuda Tarea',
            'horario_consulta': 'Horario',
            'recurso_educativo': 'Recurso',
            'estado_academico': 'Estado Académico',
            'saludo': 'Saludo',
            'despedida': 'Despedida',
            'soporte_tecnico': 'Soporte Técnico'
        };
        const sourceLabel = source === 'groq' ? '🤖 Groq LLM' : '⚙️ ML';
        intentBadge = `<span class="intent-badge-msg">${sourceLabel} | ${intentLabels[intent] || intent} (${(confidence * 100).toFixed(1)}%)</span>`;
    }
    
    msgDiv.innerHTML = `
        <div class="message-avatar">${avatarIcon}</div>
        <div class="message-content">
            <div class="message-bubble">${escapeHtml(text)}${intentBadge}</div>
            <div class="message-time">${timeStr}</div>
        </div>
    `;
    
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
}

// Mostrar indicador de escritura
function showTyping() {
    const container = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message';
    typingDiv.id = 'typingIndicator';
    typingDiv.innerHTML = `
        <div class="message-avatar"><i class="bi bi-robot"></i></div>
        <div class="message-content">
            <div class="message-bubble">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    container.appendChild(typingDiv);
    container.scrollTop = container.scrollHeight;
}

// Remover indicador de escritura
function removeTyping() {
    const typing = document.getElementById('typingIndicator');
    if (typing) typing.remove();
}

// Actualizar métricas en el sidebar
function updateMetrics(data) {
    // Intent
    const intentEl = document.getElementById('metricIntent');
    if (intentEl) intentEl.textContent = data.intent || '-';
    
    // Confidence
    const confBar = document.getElementById('metricConfidence');
    if (confBar) {
        const conf = (data.confidence || 0) * 100;
        confBar.style.width = conf + '%';
        confBar.textContent = conf.toFixed(1) + '%';
    }
    
    // Hardware
    if (data.hardware) {
        const cpu = document.getElementById('metricCPU');
        const mem = document.getElementById('metricMemory');
        const temp = document.getElementById('metricTemp');
        const net = document.getElementById('metricNetwork');
        const reqs = document.getElementById('metricRequests');
        const time = document.getElementById('metricTime');
        
        if (cpu) cpu.textContent = (data.hardware.gpu_usage || 0).toFixed(1) + '%';
        if (mem) mem.textContent = (data.hardware.memory_used_mb || 0).toFixed(0) + ' MB';
        if (temp) temp.textContent = (data.hardware.temperature_c || 0).toFixed(1) + ' °C';
        if (net) net.textContent = (data.hardware.network_latency_ms || 0).toFixed(1) + ' ms';
        if (reqs) reqs.textContent = data.request_id || '0';
        if (time) time.textContent = (data.hardware.processing_time_ms || 0).toFixed(1) + ' ms';
    }
}

// Feedback
function showFeedback() {
    const panel = document.getElementById('feedbackPanel');
    if (panel) {
        panel.style.display = 'block';
        // Reset stars
        document.querySelectorAll('.star-btn').forEach(btn => {
            btn.classList.remove('active');
            btn.querySelector('i').className = 'bi bi-star';
        });
    }
}

function closeFeedback() {
    const panel = document.getElementById('feedbackPanel');
    if (panel) panel.style.display = 'none';
}

// Star rating
document.addEventListener('DOMContentLoaded', function() {
    const stars = document.querySelectorAll('.star-btn');
    stars.forEach(star => {
        star.addEventListener('click', function() {
            const rating = parseInt(this.dataset.rating);
            
            // Visual update
            stars.forEach(s => {
                const r = parseInt(s.dataset.rating);
                if (r <= rating) {
                    s.classList.add('active');
                    s.querySelector('i').className = 'bi bi-star-fill';
                } else {
                    s.classList.remove('active');
                    s.querySelector('i').className = 'bi bi-star';
                }
            });
            
            // Send feedback
            fetch('/chat/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: document.getElementById('userId').textContent,
                    message: lastUserMessage,
                    response: lastBotResponse,
                    rating: rating
                })
            }).catch(err => console.error('Feedback error:', err));
            
            setTimeout(() => closeFeedback(), 1000);
        });
    });
    
    // Enter key
    document.getElementById('messageInput').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Toggle sidebar
    const toggleBtn = document.getElementById('toggleSidebar');
    const closeBtn = document.getElementById('closeSidebar');
    const sidebar = document.getElementById('sidebar');
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => sidebar.classList.toggle('show'));
    }
    if (closeBtn) {
        closeBtn.addEventListener('click', () => sidebar.classList.remove('show'));
    }
    
    // Clear chat
    const clearBtn = document.getElementById('clearChat');
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            const messages = document.getElementById('chatMessages');
            messages.innerHTML = `
                <div class="message bot-message">
                    <div class="message-avatar"><i class="bi bi-robot"></i></div>
                    <div class="message-content">
                        <div class="message-bubble">¡Chat reiniciado! ¿En qué puedo ayudarte?</div>
                        <div class="message-time">Ahora</div>
                    </div>
                </div>
            `;
        });
    }
    
    // Auto-refresh system uptime
    setInterval(refreshSystemMetrics, 5000);
    refreshSystemMetrics();
});

function refreshSystemMetrics() {
    fetch('/system/status')
        .then(res => res.json())
        .then(data => {
            if (data.system) {
                const uptime = document.getElementById('metricUptime');
                if (uptime) uptime.textContent = data.system.uptime_formatted || '0:00:00';
            }
        })
        .catch(() => {});
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
