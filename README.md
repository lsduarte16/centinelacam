# CAM-PI Gate Controller 🏭

**Sistema de visión inteligente para control de compuerta de andén logístico.**

Procesamiento 100% en el edge con Raspberry Pi 5, cámara domo y modelos de IA locales.

![Caja de Visión Inteligente](docs/assets/concept.png)

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                   RASPBERRY PI 5                     │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐  │
│  │  Cámara  │──▶│  YOLOv8  │──▶│ Gate Logic    │  │
│  │  RTSP    │   │  Nano    │   │ (Zonas/Track) │  │
│  └──────────┘   └──────────┘   └───────┬───────┘  │
│                                         │          │
│  ┌──────────┐   ┌──────────┐   ┌───────▼───────┐  │
│  │  Ollama  │◀──│  Eventos │◀──│  Relay GPIO   │  │
│  │ TinyLlama│   │  SQLite  │   │  (Compuerta)  │  │
│  └──────────┘   └──────────┘   └───────────────┘  │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │            FastAPI (REST API :8000)           │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
   ┌───────────┐                 ┌────────────┐
   │ Cloud S3  │                 │  Dashboard │
   │  Backup   │                 │   (Web)    │
   └───────────┘                 └────────────┘
```

## Características

- **Detección en tiempo real** con YOLOv8 Nano (personas, vehículos, camiones)
- **Tracking de objetos** con ByteTrack para trazabilidad
- **Control de compuerta** automático via GPIO relay
- **Análisis contextual** con TinyLlama/Ollama (modelo local)
- **Conteo bidireccional** personas y vehículos
- **API REST** para integración con WMS
- **Auto-deploy** desde GitHub hacia la Raspberry Pi
- **Almacenamiento local** + backup a nube (S3/GCS)

## Requisitos Hardware

| Componente | Especificación |
|---|---|
| SBC | Raspberry Pi 5 (8GB RAM) |
| Cámara | Domo IP gran angular (RTSP) |
| Relay | Módulo relay 5V/24V para compuerta |
| Alimentación | PoE HAT o fuente 24V DC |
| Storage | microSD 64GB + SSD USB (opcional) |
| Caja | IP65 para exterior |

## Instalación Rápida

### 1. Setup inicial de la Raspberry Pi

```bash
# SSH a la Raspberry
ssh pi@<IP_RASPBERRY>

# Ejecutar script de setup
curl -fsSL https://raw.githubusercontent.com/<tu-usuario>/cam-pi/main/scripts/setup_raspberry.sh | bash
```

### 2. Configurar cámara

Editar `/opt/cam-pi/config/settings.yaml`:

```yaml
camera:
  rtsp_url: "rtsp://admin:tu_password@192.168.1.100:554/stream1"
  fps: 15
  resolution: [1280, 720]
```

### 3. Configurar GPIO (relay compuerta)

```yaml
gate:
  gpio_pin: 17        # Pin BCM del relay
  open_duration: 30   # Segundos que permanece abierta
  zones:
    entry: [100, 200, 600, 500]   # Zona de entrada [x1,y1,x2,y2]
    exit: [700, 200, 1200, 500]   # Zona de salida
```

### 4. Iniciar servicio

```bash
sudo systemctl start cam-pi
sudo systemctl status cam-pi
```

## Desarrollo Local

```bash
# Clonar
git clone https://github.com/<tu-usuario>/cam-pi.git
cd cam-pi

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -e ".[dev]"

# Ejecutar tests
pytest tests/ -v

# Lint
ruff check src/
ruff format src/

# Ejecutar localmente (necesita cámara o video de prueba)
python -m src.pipeline
```

## Deploy Automático

El sistema usa un **timer systemd** que cada 5 minutos:

1. Hace `git pull` del repositorio
2. Compara si hay cambios
3. Si cambió solo código → reinicia el contenedor
4. Si cambió Dockerfile/deps → rebuild completo
5. Verifica salud → rollback automático si falla

**Flujo de trabajo:**
```
[Tu PC] → git push → [GitHub Actions CI] → [Raspberry Pi auto-pull cada 5min]
```

## API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del sistema |
| GET | `/events?since_hours=1` | Eventos recientes |
| GET | `/summary?date=2024-01-15` | Resumen diario |
| GET | `/stats` | Estadísticas en vivo |
| POST | `/gate` | Control manual (`{"action": "open"}`) |

## Modelos IA

| Modelo | Uso | Tamaño | FPS en RPi5 |
|---|---|---|---|
| YOLOv8n | Detección objetos | 6.2 MB | ~8-12 FPS |
| TinyLlama 1.1B | Análisis contextual | 637 MB | ~2-3 tok/s |
| Phi-3 mini | Alternativa SLM | 2.3 GB | ~1-2 tok/s |

## Estructura del Proyecto

```
cam-pi/
├── src/
│   ├── camera/          # Captura RTSP
│   ├── detector/        # YOLOv8 + tracking
│   ├── gate_logic/      # Control de compuerta
│   ├── llm_engine/      # Ollama/TinyLlama
│   ├── storage/         # SQLite + cloud sync
│   ├── api/             # FastAPI REST
│   └── pipeline.py      # Orquestador principal
├── config/
│   └── settings.yaml    # Configuración
├── scripts/
│   ├── setup_raspberry.sh   # Setup inicial
│   └── auto_update.sh       # Auto-deploy
├── tests/
├── docker-compose.yml
├── Dockerfile
└── .github/workflows/ci.yml
```

## Licencia

MIT
