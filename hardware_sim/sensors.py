"""
Simulación de Hardware - Módulos virtuales que emulan
el comportamiento de sensores y componentes físicos del sistema.

Esta capa simula:
- Sensor de entrada de texto (emula teclado/micrófono)
- Módulo de procesamiento NLP (emula TPU/GPU)
- Módulo de memoria (emula RAM/almacenamiento)
- Módulo de red (emula conectividad)
- Sensor de temperatura del sistema
- Módulo de alimentación (batería/fuente)
"""

import time
import random
import threading
import math
from datetime import datetime


class HardwareSensor:
    """Clase base para sensores de hardware simulados."""
    
    def __init__(self, name, unit, min_val, max_val, noise_factor=0.02):
        self.name = name
        self.unit = unit
        self.min_val = min_val
        self.max_val = max_val
        self.noise_factor = noise_factor
        self.current_value = (min_val + max_val) / 2
        self.history = []
        self.active = True
        self._lock = threading.Lock()
    
    def read(self):
        """Lee el valor actual del sensor con ruido simulado."""
        if not self.active:
            return None
        noise = random.uniform(-self.noise_factor, self.noise_factor) * self.current_value
        value = self.current_value + noise
        value = max(self.min_val, min(self.max_val, value))
        with self._lock:
            self.current_value = value
            self.history.append({
                'timestamp': datetime.now().isoformat(),
                'value': round(value, 4),
                'unit': self.unit
            })
            if len(self.history) > 1000:
                self.history = self.history[-500:]
        return round(value, 4)
    
    def set_value(self, value):
        """Establece el valor base del sensor."""
        with self._lock:
            self.current_value = max(self.min_val, min(self.max_val, value))
    
    def get_status(self):
        """Retorna el estado actual del sensor."""
        return {
            'name': self.name,
            'value': self.read(),
            'unit': self.unit,
            'active': self.active,
            'min': self.min_val,
            'max': self.max_val
        }


class TextInputSensor(HardwareSensor):
    """Simula el sensor de entrada de texto (teclado/micrófono)."""
    
    def __init__(self):
        super().__init__("Sensor de Entrada de Texto", "char/s", 0, 500, 0.05)
        self.buffer_size = 0
        self.input_rate = 0
        self.last_input_time = None
    
    def process_input(self, text):
        """Simula la recepción de entrada de texto."""
        self.buffer_size = len(text)
        self.input_rate = len(text) / max(0.1, random.uniform(0.5, 2.0))
        self.last_input_time = datetime.now().isoformat()
        self.set_value(self.input_rate)
        return {
            'buffer_size': self.buffer_size,
            'input_rate': round(self.input_rate, 2),
            'timestamp': self.last_input_time,
            'encoding': 'UTF-8',
            'channel': 'text'
        }
    
    def get_status(self):
        status = super().get_status()
        status.update({
            'buffer_size': self.buffer_size,
            'last_input': self.last_input_time,
            'channel': 'text'
        })
        return status


class NLPProcessor(HardwareSensor):
    """Simula el módulo de procesamiento NLP (TPU/GPU virtual)."""
    
    def __init__(self):
        super().__init__("Procesador NLP", "%", 0, 100, 0.01)
        self.gpu_temp = 45.0
        self.gpu_usage = 0
        self.model_loaded = False
        self.inference_count = 0
        self.total_inference_time = 0
    
    def process(self, text_length):
        """Simula el procesamiento NLP de un texto."""
        # Simular carga de procesamiento proporcional al texto
        processing_load = min(95, 20 + (text_length / 10))
        self.set_value(processing_load)
        
        # Simular temperatura GPU
        self.gpu_temp = min(85, 45 + processing_load * 0.4 + random.uniform(-2, 2))
        self.gpu_usage = processing_load
        
        # Simular tiempo de inferencia
        inference_time = 0.05 + (text_length * 0.002) + random.uniform(0, 0.03)
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        return {
            'processing_load': round(processing_load, 2),
            'gpu_temp': round(self.gpu_temp, 2),
            'gpu_usage': round(self.gpu_usage, 2),
            'inference_time_ms': round(inference_time * 1000, 2),
            'avg_inference_time_ms': round(
                (self.total_inference_time / self.inference_count) * 1000, 2
            ),
            'total_inferences': self.inference_count
        }
    
    def load_model(self):
        """Simula la carga del modelo ML."""
        time.sleep(0.5)  # Simular tiempo de carga
        self.model_loaded = True
        self.set_value(15)  # Uso base con modelo cargado
        return {'model_loaded': True, 'load_time_ms': 500}
    
    def get_status(self):
        status = super().get_status()
        status.update({
            'gpu_temp': round(self.gpu_temp, 2),
            'gpu_usage': round(self.gpu_usage, 2),
            'model_loaded': self.model_loaded,
            'inference_count': self.inference_count
        })
        return status


class MemoryModule(HardwareSensor):
    """Simula el módulo de memoria (RAM/Almacenamiento)."""
    
    def __init__(self):
        super().__init__("Módulo de Memoria", "MB", 128, 2048, 0.01)
        self.ram_used = 256
        self.ram_total = 2048
        self.storage_used = 50
        self.storage_total = 512
        self.cache_hits = 0
        self.cache_misses = 0
    
    def allocate(self, size_mb):
        """Simula la asignación de memoria."""
        self.ram_used = min(self.ram_total, self.ram_used + size_mb)
        self.set_value(self.ram_used)
        allocated = self.ram_used <= self.ram_total
        if allocated:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        return {
            'allocated': allocated,
            'ram_used': self.ram_used,
            'ram_total': self.ram_total,
            'ram_percent': round((self.ram_used / self.ram_total) * 100, 2)
        }
    
    def release(self, size_mb):
        """Simula la liberación de memoria."""
        self.ram_used = max(128, self.ram_used - size_mb)
        self.set_value(self.ram_used)
    
    def get_status(self):
        status = super().get_status()
        status.update({
            'ram_used': self.ram_used,
            'ram_total': self.ram_total,
            'ram_percent': round((self.ram_used / self.ram_total) * 100, 2),
            'storage_used': self.storage_used,
            'storage_total': self.storage_total,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_ratio': round(
                self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100, 2
            )
        })
        return status


class NetworkModule(HardwareSensor):
    """Simula el módulo de red (conectividad)."""
    
    def __init__(self):
        super().__init__("Módulo de Red", "Mbps", 0, 1000, 0.03)
        self.latency = 15
        self.packets_sent = 0
        self.packets_received = 0
        self.errors = 0
        self.connection_status = 'connected'
        self.bandwidth_used = 0
    
    def send_request(self, data_size_kb):
        """Simula el envío de una petición de red."""
        self.packets_sent += 1
        self.latency = 10 + random.uniform(0, 30)
        self.bandwidth_used = min(1000, data_size_kb * 0.1 + random.uniform(0, 50))
        self.set_value(self.bandwidth_used)
        
        # Simular pérdida de paquetes (1% probabilidad)
        if random.random() < 0.01:
            self.errors += 1
            return {'success': False, 'error': 'packet_loss', 'latency': round(self.latency, 2)}
        
        self.packets_received += 1
        return {
            'success': True,
            'latency_ms': round(self.latency, 2),
            'bandwidth_mbps': round(self.bandwidth_used, 2),
            'data_size_kb': data_size_kb
        }
    
    def get_status(self):
        status = super().get_status()
        status.update({
            'latency_ms': round(self.latency, 2),
            'connection': self.connection_status,
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'errors': self.errors,
            'packet_loss_rate': round(
                self.errors / max(1, self.packets_sent) * 100, 4
            )
        })
        return status


class TemperatureSensor(HardwareSensor):
    """Simula el sensor de temperatura del sistema."""
    
    def __init__(self):
        super().__init__("Sensor de Temperatura", "°C", 20, 95, 0.02)
        self.fan_speed = 30
        self.critical_temp = 85
        self.throttling = False
    
    def update_temperature(self, cpu_load):
        """Actualiza la temperatura basada en la carga del CPU."""
        base_temp = 35
        load_temp = cpu_load * 0.5
        ambient = random.uniform(-2, 2)
        new_temp = base_temp + load_temp + ambient
        self.set_value(new_temp)
        
        # Control automático de ventilador
        if new_temp > 60:
            self.fan_speed = min(100, 30 + (new_temp - 60) * 3)
        else:
            self.fan_speed = 30
        
        # Throttling térmico
        self.throttling = new_temp > self.critical_temp
        
        return {
            'temperature': round(new_temp, 2),
            'fan_speed': round(self.fan_speed, 2),
            'throttling': self.throttling,
            'warning': new_temp > 75
        }
    
    def get_status(self):
        status = super().get_status()
        status.update({
            'fan_speed': round(self.fan_speed, 2),
            'throttling': self.throttling,
            'critical_temp': self.critical_temp
        })
        return status


class PowerModule(HardwareSensor):
    """Simula el módulo de alimentación."""
    
    def __init__(self):
        super().__init__("Módulo de Alimentación", "%", 0, 100, 0.005)
        self.voltage = 5.0
        self.current = 0.5
        self.power_source = 'USB'
        self.uptime_seconds = 0
    
    def update(self):
        """Actualiza el estado de alimentación."""
        self.voltage = 5.0 + random.uniform(-0.1, 0.1)
        self.current = 0.5 + random.uniform(-0.05, 0.05)
        self.uptime_seconds += 1
        # Simular descarga lenta de batería
        battery = max(0, 100 - (self.uptime_seconds / 3600))
        self.set_value(battery)
        return {
            'voltage': round(self.voltage, 3),
            'current_A': round(self.current, 3),
            'power_W': round(self.voltage * self.current, 3),
            'battery_percent': round(battery, 2),
            'source': self.power_source,
            'uptime_s': self.uptime_seconds
        }
    
    def get_status(self):
        status = super().get_status()
        status.update({
            'voltage': round(self.voltage, 3),
            'current_A': round(self.current, 3),
            'power_W': round(self.voltage * self.current, 3),
            'source': self.power_source,
            'uptime_s': self.uptime_seconds
        })
        return status


class HardwareSimulator:
    """Orquestador principal del simulador de hardware."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        # Inicializar todos los módulos de hardware
        self.text_input = TextInputSensor()
        self.nlp_processor = NLPProcessor()
        self.memory = MemoryModule()
        self.network = NetworkModule()
        self.temperature = TemperatureSensor()
        self.power = PowerModule()
        
        self.start_time = datetime.now()
        self.request_count = 0
        self.error_count = 0
        
        # Cargar modelo ML simulado
        self.nlp_processor.load_model()
        
        # Hilo de monitoreo
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _monitor_loop(self):
        """Bucle de monitoreo del hardware simulado."""
        while self._monitoring:
            # Actualizar temperatura basada en carga del procesador
            self.temperature.update_temperature(self.nlp_processor.gpu_usage)
            self.power.update()
            time.sleep(2)
    
    def process_chat_input(self, text):
        """Procesa una entrada de chat a través de todos los módulos de hardware."""
        self.request_count += 1
        
        # 1. Entrada de texto
        input_status = self.text_input.process_input(text)
        
        # 2. Asignación de memoria para procesamiento
        mem_needed = max(10, len(text) * 0.5)
        mem_status = self.memory.allocate(mem_needed)
        
        # 3. Procesamiento NLP
        nlp_status = self.nlp_processor.process(len(text))
        
        # 4. Comunicación de red (respuesta)
        net_status = self.network.send_request(len(text) * 0.1)
        
        # 5. Liberar memoria después del procesamiento
        self.memory.release(mem_needed * 0.7)
        
        # 6. Verificar temperatura
        temp_status = self.temperature.update_temperature(self.nlp_processor.gpu_usage)
        
        return {
            'request_id': self.request_count,
            'input': input_status,
            'memory': mem_status,
            'nlp': nlp_status,
            'network': net_status,
            'temperature': temp_status,
            'power': self.power.update(),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_full_status(self):
        """Retorna el estado completo de todo el hardware simulado."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            'system': {
                'uptime_seconds': round(uptime, 2),
                'uptime_formatted': self._format_uptime(uptime),
                'total_requests': self.request_count,
                'error_count': self.error_count,
                'status': 'operational' if not self.temperature.throttling else 'throttled'
            },
            'sensors': {
                'text_input': self.text_input.get_status(),
                'nlp_processor': self.nlp_processor.get_status(),
                'memory': self.memory.get_status(),
                'network': self.network.get_status(),
                'temperature': self.temperature.get_status(),
                'power': self.power.get_status()
            }
        }
    
    def _format_uptime(self, seconds):
        """Formatea el tiempo de actividad."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def shutdown(self):
        """Apaga el simulador de hardware."""
        self._monitoring = False


# Instancia global del simulador
hardware = HardwareSimulator()
