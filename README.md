# 🕵️‍♂️ Bot Forense de Voz (Telegram)

Este bot analiza archivos MP3 y genera un informe técnico con biometría de voz, detección de IA, similitud entre voces y metadatos. Diseñado para su uso en investigaciones forenses, análisis pericial y detección de deepfakes.

## 🚀 Funcionalidades

- 🎧 **Análisis individual de voz**
  - Pitch medio (frecuencia fundamental)
  - ZCR (Zero Crossing Rate)
  - Entropía espectral
  - Varianza de MFCCs
  - Planitud espectral
  - Análisis de banda de frecuencias altas
  - Estimación de género (masculina o femenina)
  - Detección de voz artificial (IA)

- 🗣️ **Similitud entre voces**
  - Segmentación automática del audio en voces
  - Generación de embeddings de voz
  - Cálculo de similitud mediante distancia coseno

- 📊 **Informe completo**
  - Análisis biométrico detallado
  - Lectura de metadatos del MP3
  - Informe en Telgram

## 🧠 Requisitos

- Python 3.8 o superior
- Recomendado: entorno virtual (venv)

## 📦 Instalación

Clona el repositorio y crea un entorno virtual:

```bash
git clone [https://github.com/tu_usuario/bot-forense-voz.git](https://github.com/Quantika14/forensic-voice-telegram-bot/)
cd bot-forense-voz
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt```

- ⚙️ ** Configuración **
Crea tu bot en BotFather y copia el token.

Sustituye la constante TOKEN = 'TU_TOKEN_AQUI' en el archivo bot.py.

Los modelos de speechbrain se descargarán automáticamente al primer uso.

- 👤 ** Autor** 
Jorge Coronado
Quantika14 – Investigación Digital y Ciberseguridad
📧 jorge.coronado@quantika14.com
🔗 LinkedIn
🌐 https://quantika14.com

- 🛡️ ** Licencia **
Este proyecto está licenciado bajo los términos de la GNU General Public License v3.0.
Consulta el archivo LICENSE para más detalles.
