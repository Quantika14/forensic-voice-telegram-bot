# --- IMPORTACIONES ---
import torch
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier
from datetime import datetime
import time
import json
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import os
from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import webrtcvad
import contextlib
import wave

#AUTOR JORGE CORONADO 
#EMPRESA QUANTIKA14
#VERSION 1 BETA 

# --- CONFIGURACIÓN INICIAL ---
TOKEN = 'TOKEN AQUI'
DEBUG_FILE = 'debug_config.json'

def contar_segmentos_de_voz(signal, sr, frame_ms=30):
    """
    Cuenta el número de segmentos de voz y silencios usando WebRTC VAD.
    """
    import struct
    vad = webrtcvad.Vad(2)  # agresividad 0–3
    frame_size = int(sr * frame_ms / 1000)
    num_frames = len(signal) // frame_size
    voiced_flags = []
    
    for i in range(num_frames):
        frame = signal[i * frame_size: (i + 1) * frame_size]
        if len(frame) < frame_size:
            break
        pcm = (frame * 32768).astype(np.int16).tobytes()
        is_voiced = vad.is_speech(pcm, sr)
        voiced_flags.append(is_voiced)

    total_voiced = sum(voiced_flags)
    total_silence = len(voiced_flags) - total_voiced
    speech_ratio = total_voiced / len(voiced_flags)

    return {
        "Frames con voz": total_voiced,
        "Frames sin voz": total_silence,
        "Ratio voz/silencio": round(speech_ratio, 3)
    }


# --- INICIALIZAR CLASIFICADOR DE VOZ ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/spkrec-ecapa-voxceleb"
).to(device)

# --- CONFIGURACIÓN DE DEPURACIÓN ---
def load_debug_config():
    try:
        with open(DEBUG_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {
            "pitch_gender_threshold": 160,
            "ai_mfcc_variance_threshold": 50,
            "similarity_threshold": 0.75
        }

# --- CARGA DE AUDIO Y EMBEDDING ---
def load_audio(path, target_sr=16000):
    sig, sr = librosa.load(path, sr=target_sr, mono=True)
    sig_tensor = torch.tensor(sig).unsqueeze(0).to(device)
    return sig, sig_tensor, sr

def get_embedding_mp3(path):
    signal, tensor_signal, sr = load_audio(path)
    embedding = classifier.encode_batch(tensor_signal)
    embedding = embedding.squeeze().cpu().numpy()
    return embedding / np.linalg.norm(embedding), signal, sr

def voice_similarity(e1, e2):
    return float(1 - cosine(e1, e2))

# --- METADATOS DEL MP3 ---
def extract_metadata(path):
    try:
        audio = MP3(path, ID3=EasyID3)
        metadata = {
            "Título": audio.get("title", ["Desconocido"])[0],
            "Artista": audio.get("artist", ["Desconocido"])[0],
            "Álbum": audio.get("album", ["Desconocido"])[0],
            "Duración (metadatos)": round(audio.info.length, 2),
            "Bitrate": audio.info.bitrate
        }
        return metadata
    except Exception:
        return {"Metadatos": "No disponibles o corruptos"}

# --- EXTRACCIÓN DE DATOS BIOMÉTRICOS ---
def extract_biometric_features(signal, sr):
    config = load_debug_config()
    pitch_thresh = config.get("pitch_gender_threshold", 160)
    var_thresh = config.get("ai_mfcc_variance_threshold", 50)
    max_duration = 120.0

    duration = librosa.get_duration(y=signal, sr=sr)
    if duration > max_duration:
        return {"error": f"Audio demasiado largo ({duration:.2f} s). Máximo permitido: {max_duration} segundos."}

    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
        spectral_entropy = -np.sum((mfccs/np.sum(mfccs)) * np.log2((mfccs+1e-8)/np.sum(mfccs)), axis=0).mean()
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=signal))
        high_freq_energy = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr, roll_percent=0.99))

        f0, voiced_flag, _ = librosa.pyin(signal, fmin=75, fmax=350, sr=sr)
        voiced_f0 = f0[~np.isnan(f0)]
        avg_pitch = float(round(np.mean(voiced_f0), 2)) if voiced_f0.size > 0 else 'No detectado'
        pitch_std = float(np.std(voiced_f0)) if voiced_f0.size > 0 else 0.0
        gender = 'Femenina' if isinstance(avg_pitch, float) and avg_pitch > pitch_thresh else 'Masculina'

        # Heurísticas para IA
        formant_anomaly = avg_pitch == 'No detectado' or avg_pitch < 75 or avg_pitch > 350
        low_mfcc_var = np.var(mfccs) < var_thresh
        high_entropy = spectral_entropy < 3.0
        unnatural_zcr = zcr < 0.01 or zcr > 0.25
        unnatural_bandwidth = bandwidth < 500 or bandwidth > 4000
        flatness_issue = spectral_flatness > 0.35
        low_pitch_var = pitch_std < 10.0
        no_high_freq = high_freq_energy < 7000

        ia_score = sum([
            low_mfcc_var,
            high_entropy,
            unnatural_zcr,
            unnatural_bandwidth,
            formant_anomaly,
            flatness_issue,
            low_pitch_var,
            no_high_freq
        ])

        ai_generated = 'Posible voz IA' if ia_score >= 4 else 'Voz natural probable'

        debug_data = {
            "Pitch medio": avg_pitch,
            "Desviación Pitch": round(pitch_std, 2),
            "Género detectado": gender,
            "Varianza MFCC": float(np.var(mfccs).round(3)),
            "Entropía espectral": round(spectral_entropy, 3),
            "ZCR": round(zcr, 3),
            "Bandwidth": round(bandwidth, 2),
            "Planitud espectral": round(spectral_flatness, 3),
            "Rolloff 99%": round(high_freq_energy, 2),
            "Formante anómalo": formant_anomaly,
            "Origen detectado": ai_generated
        }
        # Análisis rítmico
        ritmo = contar_segmentos_de_voz(signal, sr)
        speech_ratio = ritmo["Ratio voz/silencio"]

        ritmo_anómalo = speech_ratio > 0.95 or speech_ratio < 0.25  # demasiada o muy poca voz

        ia_score += int(ritmo_anómalo)

        debug_data.update({
            "Frames con voz": ritmo["Frames con voz"],
            "Frames sin voz": ritmo["Frames sin voz"],
            "Ratio voz/silencio": speech_ratio,
        })

        return {
            "Duración (segundos)": float(round(duration, 2)),
            "Media MFCC": [float(x) for x in np.mean(mfccs, axis=1).round(3)],
            "Varianza MFCC": [float(x) for x in np.var(mfccs, axis=1).round(3)],
            **debug_data
        }

    except Exception as e:
        return {"error": f"Error en el análisis: {str(e)}"}

# --- INFORMES ---
def analizar_audio_individual(path):
    embedding, signal, sr = get_embedding_mp3(path)
    biometric = extract_biometric_features(signal, sr)
    metadata = extract_metadata(path)

    if 'error' in biometric:
        return f"❌ Error: {biometric['error']}"

    report = f"📌 *Informe de Análisis de Audio*\n"
    report += f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "🎧 *Metadatos:*\n"
    for k, v in metadata.items():
        report += f"- {k}: {v}\n"
    report += "\n🎙 *Análisis Biométrico:*\n"
    for k, v in biometric.items():
        report += f"- {k}: {v}\n"

    report += "\n📘 *Explicación de los conceptos:*\n"
    report += "- *Pitch medio:* indica la frecuencia fundamental, útil para diferenciar voces masculinas y femeninas.\n"
    report += "- *ZCR (Zero Crossing Rate):* mide el número de veces que la señal cruza el eje cero; voces IA suelen tener ZCRs atípicos.\n"
    report += "- *Entropía espectral:* mide la impredecibilidad de la señal; voces artificiales suelen tener menor entropía.\n"
    report += "- *MFCCs:* parámetros que resumen el timbre. Su varianza baja puede indicar voz sintetizada.\n"
    return report

def detectar_segmentos_de_voz(signal, sr, min_silence_len=0.5, threshold=20):
    intervals = librosa.effects.split(signal, top_db=threshold)
    segmentos = []
    for i, (start, end) in enumerate(intervals):
        if librosa.get_duration(y=signal[start:end], sr=sr) > 2.5:
            segmentos.append((start, end))
    return segmentos

def analizar_similitud_interna(path):
    embedding_list = []
    report = f"📌 *Similitud de Voces Interna*\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    signal, tensor_signal, sr = load_audio(path)
    segments = detectar_segmentos_de_voz(signal, sr)

    if len(segments) < 2:
        return "❌ No se detectaron múltiples segmentos de voz claramente diferenciados."

    voices = []
    for i, (start, end) in enumerate(segments):
        seg = signal[start:end]
        seg_tensor = torch.tensor(seg).unsqueeze(0).to(device)
        emb = classifier.encode_batch(seg_tensor).squeeze().cpu().numpy()
        emb /= np.linalg.norm(emb)
        bio = extract_biometric_features(seg, sr)
        voices.append((f"Voz {i+1}", emb, bio))

    for i in range(len(voices)):
        nombre_i, emb_i, bio_i = voices[i]
        report += f"🔹 {nombre_i}\n"
        for k, v in bio_i.items():
            report += f"- {k}: {v}\n"
        report += "\n"

    report += "\n📊 *Comparaciones de Similitud:*\n"
    for i in range(len(voices)):
        for j in range(i+1, len(voices)):
            sim = voice_similarity(voices[i][1], voices[j][1])
            report += f"{voices[i][0]} vs {voices[j][0]}: {round(sim, 3)}\n"

    return report

# --- COMANDOS TELEGRAM ---
async def elegir_analisis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    teclado = ReplyKeyboardMarkup([["/analizar_audio"], ["/similitud_voces"]], resize_keyboard=True)
    await update.message.reply_text("Selecciona una opción para comenzar:", reply_markup=teclado)

async def analizar_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['modo'] = 'analizar'
    context.user_data['audios'] = []
    await update.message.reply_text("🎧 Sube un archivo MP3 para analizarlo.")

async def similitud_voces(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['modo'] = 'similitud'
    context.user_data['audios'] = []
    await update.message.reply_text("🎤 Sube un archivo MP3. Intentaré detectar y comparar varias voces.")

async def autor(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👨‍💻 *Autor del sistema de análisis de voz:* Jorge Coronado\n\n"
        "📧 Email: jorge.coronado@quantika14.com\n"
        "📱 Instagram: [@elperitoinf](https://instagram.com/elperitoinf)\n"
        "🔗 LinkedIn: [Perfil](https://es.linkedin.com/in/jorge-coronado-quantika14)"
    )
    await update.message.reply_text(msg, parse_mode='Markdown')

async def handle_files(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = await update.message.audio.get_file()
    file_path = f"{file.file_id}.mp3"
    await file.download_to_drive(file_path)

    modo = context.user_data.get('modo')

    if modo == 'analizar':
        await update.message.reply_text("🔎 Analizando audio individual...")
        report = analizar_audio_individual(file_path)
        await update.message.reply_text(report, parse_mode='Markdown')
        os.remove(file_path)

    elif modo == 'similitud':
        await update.message.reply_text("🔎 Detectando y comparando voces en el audio...")
        report = analizar_similitud_interna(file_path)
        await update.message.reply_text(report, parse_mode='Markdown')
        os.remove(file_path)

    else:
        await update.message.reply_text("❗ Por favor selecciona primero una opción usando /start.")

# --- INICIAR BOT ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler('start', elegir_analisis))
    app.add_handler(CommandHandler('autor', autor))
    app.add_handler(CommandHandler('analizar_audio', analizar_audio))
    app.add_handler(CommandHandler('similitud_voces', similitud_voces))
    app.add_handler(MessageHandler(filters.AUDIO, handle_files))
    print("Bot iniciado...")
    app.run_polling()
