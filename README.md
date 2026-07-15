# CentinelaCam - Edge AI Vision System

**Sistema de visión inteligente en el edge con Raspberry Pi 5.**

Procesamiento 100% local con modelos de IA (YOLOv8, Ollama), notificaciones Telegram y casos de uso configurables.

---

## Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│                     RASPBERRY PI 5                         │
│                                                          │
│  ┌───────────┐   ┌───────────┐   ┌────────────────┐     │
│  │  Cámara   │──▶│  YOLOv8n  │──▶│  Use Case      │     │
│  │ USB/RTSP  │   │  + HSV    │   │  Engine        │     │
│  └───────────┘   └───────────┘   └───────┬────────┘     │
│                                           │              │
│  ┌───────────┐   ┌───────────┐   ┌───────▼────────┐     │
│  │  Ollama   │◀──│  SQLite   │◀──│  Event Logic   │     │
│  │ TinyLlama │   │  Events   │   │  + GPIO Relay  │     │
│  └───────────┘   └───────────┘   └───────┬────────┘     │
│                                           │              │
│  ┌────────────────────────────────────────▼───────────┐  │
│  │           FastAPI + MJPEG Stream (:8000)            │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
         │                │                    │
         ▼                ▼                    ▼
   ┌──────────┐    ┌───────────┐       ┌────────────┐
   │ Telegram │    │ Cloud S3  │       │ Dashboard  │
   │  Alerts  │    │  Backup   │       │   (Web)    │
   └──────────┘    └───────────┘       └────────────┘
```

## Casos de Uso Disponibles

Cada Raspberry Pi ejecuta **un solo caso de uso**, configurable en `settings.yaml`:

| Caso de Uso | Descripción | Detección |
|---|---|---|
| `zone_violation` | Alerta cuando objeto sale de zona segura | HSV saturación + contornos |
| `gate_control` | Control automático de compuerta de andén | YOLOv8 (personas/vehículos) |
| `people_counter` | Conteo bidireccional de personas | YOLOv8 + línea de cruce |
| `package_counter` | Conteo de paquetes en cinta | YOLOv8 + línea de cruce |
| `barcode_reader` | Lectura de códigos de barras | pyzbar + ROI |
| `sorter_monitor` | Detección de atascos en sorter | YOLOv8 + movimiento |

## Caso de Uso Activo: `zone_violation`

Demo actual: hoja blanca con dibujo de casa + figurita de color.

**Lógica:**
1. Detecta objetos de color (alta saturación HSV) dentro del área del papel (`paper_roi`)
2. Si el objeto está dentro de la zona segura (la casa) → OK (recuadro verde)
3. Si el objeto sale de la zona segura → ALERTA (recuadro rojo + notificación Telegram con foto)

**Parámetros ajustables en `settings.yaml`:**
```yaml
zone_violation:
  min_object_area: 1200        # área mínima en px (filtra ruido)
  zones:
    paper_roi: [130, 60, 950, 570]     # dónde buscar (solo el papel)
    safe_zone: [340, 180, 740, 500]    # la "casa" (zona permitida)
```

## Características

- **Detección en tiempo real** con YOLOv8 Nano o detección por color (HSV)
- **Tracking de objetos** con ByteTrack
- **Notificaciones Telegram** con snapshot de evidencia
- **Control de compuerta** automático via GPIO relay
- **Análisis contextual** con TinyLlama/Ollama (modelo local)
- **Video en vivo** MJPEG con overlays (zonas, bounding boxes, estado)
- **API REST** para integración con WMS
- **Auto-deploy** desde GitHub hacia la Raspberry Pi
- **Almacenamiento local** SQLite + backup a nube (S3/GCS)
- **FPS adaptativo** configurable por caso de uso

## Requisitos Hardware

| Componente | Especificación |
|---|---|
| SBC | Raspberry Pi 5 (8GB RAM) |
| Cámara | USB (Netum) o Domo IP RTSP |
| Relay | Módulo relay 5V/24V (para gate_control) |
| Alimentación | PoE HAT o fuente 24V DC |
| Storage | microSD 64GB |

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Detección | YOLOv8 Nano (PyTorch CPU) + OpenCV HSV |
| Tracking | ByteTrack |
| LLM Local | Ollama + TinyLlama 1.1B |
| Backend | FastAPI + Uvicorn |
| DB | SQLite (eventos + trazabilidad) |
| Notificaciones | Telegram Bot API |
| Contenedor | Docker + Docker Compose (ARM64) |
| CI/CD | GitHub Actions + systemd timer auto-pull |
| Cloud Sync | rclone (S3/GCS) |

## Instalación

### 1. Setup Raspberry Pi

```bash
ssh cxp@<IP_RASPBERRY>
curl -fsSL https://raw.githubusercontent.com/lsduarte16/centinelacam/main/scripts/setup_raspberry.sh | bash
```

### 2. Configurar `.env`

```bash
# En /opt/cam-pi/.env
TELEGRAM_BOT_TOKEN=tu_token_aqui
TELEGRAM_CHAT_ID=tu_chat_id
CAMERA_SOURCE=0   # 0=USB, o rtsp://...
```

### 3. Configurar caso de uso

Editar `/opt/cam-pi/config/settings.yaml`:

```yaml
use_case: "zone_violation"   # o gate_control, people_counter, etc.
```

### 4. Iniciar

```bash
cd /opt/cam-pi
docker compose up -d
```

### 5. Verificar

- Dashboard: `http://<IP_RASPBERRY>:8000/`
- Health: `http://<IP_RASPBERRY>:8000/health`
- Snapshot: `http://<IP_RASPBERRY>:8000/snapshot`

## Desarrollo Local

```bash
git clone https://github.com/lsduarte16/centinelacam.git
cd centinelacam

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Tests
pytest tests/ -v

# Lint
ruff check src/
ruff format src/

# Ejecutar localmente
python -m src
```

## Deploy Automático

El sistema usa un **timer systemd** que cada 5 minutos:

1. `git pull` del repositorio
2. Compara si hay cambios
3. Si cambió código → rebuild + restart contenedor
4. Verifica salud → rollback automático si falla

```
[Tu PC] → git push → [GitHub Actions CI] → [RPi auto-pull cada 5min]
```

O deploy manual desde tu PC:
```bash
ssh cxp@192.168.1.18 "cd /opt/cam-pi && git pull && docker compose up -d --build"
```

## API Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/` | Dashboard web con video en vivo |
| GET | `/health` | Estado del sistema |
| GET | `/video` | Stream MJPEG con overlays |
| GET | `/snapshot` | Captura JPEG actual |
| GET | `/events?since_hours=1` | Eventos recientes |
| GET | `/summary?date=2024-01-15` | Resumen diario |
| GET | `/stats` | Estadísticas en vivo |
| POST | `/gate` | Control manual (`{"action": "open"}`) |

## Modelos IA

| Modelo | Uso | Tamaño | Rendimiento RPi5 |
|---|---|---|---|
| YOLOv8n | Detección objetos COCO | 6.2 MB | ~8-12 FPS |
| HSV+Contornos | Detección por color | 0 MB | ~30+ FPS |
| TinyLlama 1.1B | Análisis contextual | 637 MB | ~2-3 tok/s |

## Estructura del Proyecto

```
cam-pi/
├── src/
│   ├── api/               # FastAPI + video stream
│   │   └── server.py
│   ├── camera/            # Captura USB/RTSP con auto-reconnect
│   │   └── capture.py
│   ├── detector/          # YOLOv8 + ByteTrack
│   │   ├── models.py
│   │   └── yolo_detector.py
│   ├── gate_logic/        # Control de compuerta + eventos
│   │   ├── controller.py
│   │   └── events.py
│   ├── llm_engine/        # Ollama/TinyLlama
│   │   └── analyzer.py
│   ├── notifications/     # Telegram bot
│   │   └── telegram.py
│   ├── storage/           # SQLite + cloud sync
│   │   ├── database.py
│   │   └── sync.py
│   ├── config.py          # Pydantic settings (YAML)
│   └── pipeline.py        # Orquestador principal
├── config/
│   └── settings.yaml      # Configuración centralizada
├── scripts/
│   ├── setup_raspberry.sh # Setup inicial RPi
│   └── auto_update.sh     # Auto-deploy script
├── tests/
│   └── test_gate_logic.py
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
└── README.md
```

## Configuración de Nodo

```yaml
# config/settings.yaml
node:
  id: "retail001"
  location: "Demo Lab"
  description: "Detección de figura fuera de zona"

camera:
  source: 0                    # 0=USB, o "rtsp://..."
  resolution: [1280, 720]
  reconnect_delay: 5

telegram:
  enabled: true
  cooldown: 10                 # segundos entre notificaciones
```

## Calibración de Zonas (zone_violation)

1. Abrir `http://<IP>:8000/` para ver video en vivo
2. El rectángulo gris muestra el `paper_roi` (área de búsqueda)
3. El rectángulo verde muestra la `safe_zone` (zona permitida)
4. Ajustar coordenadas `[x1, y1, x2, y2]` en `settings.yaml`
5. Redeploy: `docker compose up -d --build`

Resolución de referencia: 1280x720 px (x: 0→1280, y: 0→720)

## Pendientes / Roadmap

- [ ] Calibración de zonas desde la UI web (drag & drop)
- [ ] Histórico de eventos con fotos en dashboard
- [ ] Soporte multi-cámara por nodo
- [ ] Entrenamiento custom de YOLO para objetos específicos
- [ ] Integración WMS via webhooks
- [ ] Alertas por email además de Telegram
- [ ] Panel centralizado multi-nodo

## Licencia

MIT
