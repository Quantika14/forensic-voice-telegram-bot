# 🕵️‍♂️ Bot Forense de Voz (Telegram)

**Bot Forense de Voz** es una herramienta de análisis automático de archivos MP3 diseñada para detectar similitudes entre voces, estimar características biométricas del hablante y detectar posibles voces generadas por inteligencia artificial (IA). Ideal para peritajes forenses, investigaciones de fraude de voz o verificación de identidad en procedimientos judiciales.

---

## 🚀 Funcionalidades

### 🎧 Análisis individual de voz

- Pitch medio (frecuencia fundamental)
- ZCR (Zero Crossing Rate)
- Entropía espectral
- Varianza de MFCCs
- Planitud espectral
- Análisis de frecuencias altas
- Estimación del género (masculino o femenino)
- Detección de voz generada por IA

### 🗣️ Similitud entre voces

- Segmentación automática de audio en voces distintas
- Extracción de embeddings vocales (con SpeechBrain)
- Cálculo de similitud mediante distancia coseno
- Comparaciones cruzadas con informe técnico

### 📊 Informe detallado

- Lectura de metadatos del archivo MP3 (título, autor, duración, bitrate…)
- Informe biométrico y de autenticidad
- Formato claro y estructurado para presentación legal o técnica
- Generado automáticamente vía Telegram

---

## 🧠 Requisitos

- Python 3.8 o superior
- Recomendado: entorno virtual (venv)

---

## 📦 Instalación

git clone https://github.com/Quantika14/forensic-voice-telegram-bot.git
cd forensic-voice-telegram-bot
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

## Ejemplo de informe
📌 *Informe de Análisis de Audio*
📅 Fecha: 2025-06-25 19:45:03

🎧 *Metadatos:*
- Título: Desconocido
- Artista: Desconocido
- Álbum: Desconocido
- Duración (metadatos): 18.43
- Bitrate: 128000

🎙 *Análisis Biométrico:*
- Pitch medio: 134.5 Hz
- Desviación Pitch: 9.3
- Género detectado: Masculina
- Varianza MFCC: 67.412
- Entropía espectral: 3.42
- ZCR: 0.057
- Bandwidth: 2854.22
- Planitud espectral: 0.211
- Rolloff 99%: 10123.34
- Ratio voz/silencio: 0.71
- Origen detectado: Voz natural probable
  
# 👤 Autor
Jorge Coronado
Perito informático y experto en ciberinteligencia
Quantika14 – Investigación Digital y Ciberseguridad
📧 jorge.coronado@quantika14.com
🔗 LinkedIn
🌐 https://quantika14.com

# 🛡️ Licencia
Este proyecto está licenciado bajo los términos de la GNU General Public License v3.0.

Puedes consultar el archivo LICENSE para más detalles.

© Quantika14 – Todos los derechos reservados bajo GNU GPL v3
