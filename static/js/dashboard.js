/**
 * Chatbot Académico UPN - Dashboard JavaScript
 * Actualización automática de métricas del hardware simulado.
 */

function refreshDashboard() {
    fetch('/system/status')
        .then(res => res.json())
        .then(data => {
            updateSystemInfo(data);
            updateSensors(data.sensors);
        })
        .catch(err => console.error('Error refreshing:', err));
    
    fetch('/system/model')
        .then(res => res.json())
        .then(data => {
            // Model metrics are static, no need to update
        })
        .catch(err => console.error('Error model:', err));
}

function updateSystemInfo(data) {
    if (!data.system) return;
    
    const uptime = document.getElementById('statUptime');
    const requests = document.getElementById('statRequests');
    const badge = document.getElementById('systemStatusBadge');
    
    if (uptime) uptime.textContent = data.system.uptime_formatted || '-';
    if (requests) requests.textContent = data.system.total_requests || '0';
    
    if (badge) {
        if (data.system.status === 'throttled') {
            badge.className = 'badge bg-warning px-3 py-2';
            badge.innerHTML = '<i class="bi bi-exclamation-triangle me-1"></i>Throttled';
        } else {
            badge.className = 'badge bg-success px-3 py-2';
            badge.innerHTML = '<i class="bi bi-circle-fill me-1" style="font-size:8px"></i>Operativo';
        }
    }
}

function updateSensors(sensors) {
    if (!sensors) return;
    
    // Text Input
    if (sensors.text_input) {
        const s = sensors.text_input;
        updateEl('sensorBuffer', s.buffer_size + ' chars');
        updateEl('sensorInputRate', s.value.toFixed(1) + ' char/s');
        updateEl('sensorChannel', s.channel);
    }
    
    // NLP Processor
    if (sensors.nlp_processor) {
        const s = sensors.nlp_processor;
        updateProgressBar('sensorGPU', s.gpu_usage, s.gpu_usage.toFixed(1) + '%');
        updateEl('sensorGPUTemp', s.gpu_temp.toFixed(1) + ' °C');
        updateEl('sensorInferences', s.inference_count);
    }
    
    // Memory
    if (sensors.memory) {
        const s = sensors.memory;
        updateProgressBar('sensorRAM', s.ram_percent, 
            s.ram_used + ' / ' + s.ram_total + ' MB');
        updateEl('sensorCacheHit', s.cache_hit_ratio + '%');
        updateEl('sensorStorage', s.storage_used + ' / ' + s.storage_total + ' MB');
    }
    
    // Network
    if (sensors.network) {
        const s = sensors.network;
        updateEl('sensorLatency', s.latency_ms.toFixed(1) + ' ms');
        updateEl('sensorPacketsSent', s.packets_sent);
        updateEl('sensorPacketLoss', s.packet_loss_rate + '%');
    }
    
    // Temperature
    if (sensors.temperature) {
        const s = sensors.temperature;
        updateEl('sensorTemp', s.value.toFixed(1) + ' °C');
        updateEl('sensorFan', s.fan_speed.toFixed(0) + '%');
        updateEl('sensorThrottle', s.throttling ? 'Sí' : 'No');
        
        // Update stat card
        const statTemp = document.getElementById('statTemp');
        if (statTemp) statTemp.textContent = s.value.toFixed(1) + ' °C';
        
        // Update temp badge
        const tempBadge = document.getElementById('tempBadge');
        if (tempBadge) {
            if (s.value > 75) {
                tempBadge.className = 'badge bg-danger';
                tempBadge.textContent = 'Alta';
            } else if (s.value > 60) {
                tempBadge.className = 'badge bg-warning';
                tempBadge.textContent = 'Media';
            } else {
                tempBadge.className = 'badge bg-success';
                tempBadge.textContent = 'Normal';
            }
        }
    }
    
    // Power
    if (sensors.power) {
        const s = sensors.power;
        updateEl('sensorVoltage', s.voltage.toFixed(3) + ' V');
        updateEl('sensorCurrent', s.current_A.toFixed(3) + ' A');
        updateEl('sensorPower', s.power_W.toFixed(3) + ' W');
    }
}

function updateEl(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function updateProgressBar(id, percent, label) {
    const bar = document.getElementById(id);
    if (bar) {
        bar.style.width = percent + '%';
        bar.textContent = label;
    }
}

// Auto-refresh cada 3 segundos
document.addEventListener('DOMContentLoaded', function() {
    refreshDashboard();
    setInterval(refreshDashboard, 3000);
});
