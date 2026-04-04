"""
SmartVoice - Web Interface
Production-ready UI for the agent-based audio processing system.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys
import os
import uuid
import json
import mimetypes
import numpy as np
import time
import shutil
import glob

from flask import Flask, jsonify, render_template, request, send_file

# Allow running as a script from the workspace root without package install.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from HamSabaqAi.SmartVoice.agents import SmartVoicePipeline, OUTPUT_DIR, TEMP_DIR, DATA_DIR, ALLOWED_AUDIO, MAX_AUDIO_MB
    from HamSabaqAi.SmartVoice.agents import SemanticSearchEngine, VoiceprintStore, VoiceprintEmbedder
    from HamSabaqAi.SmartVoice.storage import SessionStore, SESSIONS_DIR, EMBEDDINGS_DIR
except ModuleNotFoundError:
    from SmartVoice.agents import SmartVoicePipeline, OUTPUT_DIR, TEMP_DIR, DATA_DIR, ALLOWED_AUDIO, MAX_AUDIO_MB
    from SmartVoice.agents import SemanticSearchEngine, VoiceprintStore, VoiceprintEmbedder
    from SmartVoice.storage import SessionStore, SESSIONS_DIR, EMBEDDINGS_DIR

# Supported languages
LANGUAGES = ['auto']


def format_transcript(segments: list) -> str:
    """Format transcript with speaker labels."""
    if not segments:
        return "No transcript available."

    lines, current_speaker = [], None
    for seg in segments:
        if seg['speaker'] != current_speaker:
            current_speaker = seg['speaker']
            lines.append(f"\n[{current_speaker}]")
        lines.append(f"  [{seg['start']:.1f}s] {seg['text']}")
    return "\n".join(lines)


def format_summary(bullets: list, keywords: list) -> Dict[str, Any]:
    """Format summary for UI."""
    return {
        "bullets": bullets or [],
        "keywords": keywords or [],
    }


def process_audio(
    audio_path: Optional[str],
    model_size: str,
    language: str,
    enable_diarization: bool,
    enable_noise_reduction: bool
) -> Dict[str, Any]:
    """Process audio through the agent pipeline and return structured results."""
    if not audio_path:
        return {"error": "Please record or upload audio first."}

    pipeline = SmartVoicePipeline()
    result = pipeline.process(
        audio_path=audio_path,
        model_size=model_size,
        language=language,
        enable_diarization=enable_diarization,
        enable_noise_reduction=enable_noise_reduction
    )

    if result.get("error"):
        return {"error": result["error"]}

    transcript = format_transcript(result.get("segments", []))
    summary = format_summary(result.get("bullet_points", []), result.get("keywords", []))
    metadata = {
        "session_id": result.get("session_id"),
        "language": result.get("detected_language"),
        "segments": len(result.get("segments", [])),
        "speakers": len(result.get("speaker_info", {})) or 1,
        "bullets": len(result.get("bullet_points", [])),
    }
    tts_path = result.get("tts_audio_path") or ""

    return {
        "status": "complete",
        "transcript": transcript,
        "summary": summary,
        "metadata": metadata,
        "segments": result.get("segments", []),
        "tts_audio_path": tts_path,
        "diarization_error": result.get("diarization_error"),
        "speaker_attribution_error": result.get("speaker_attribution_error"),
    }


def _validate_audio(file_path: Path) -> Optional[str]:
    if not file_path.exists():
        return f"File not found: {file_path}"
    if file_path.suffix.lower() not in ALLOWED_AUDIO:
        return f"Invalid format: {file_path.suffix}"
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_AUDIO_MB:
        return f"File too large (max {MAX_AUDIO_MB}MB)"
    return None


def _save_upload(uploaded) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.filename).suffix.lower()
    name = f"audio_{uuid.uuid4().hex[:8]}{suffix}"
    target = DATA_DIR / name
    uploaded.save(target)
    return target


def create_app() -> Flask:
    """Create the Flask interface."""
    app = Flask(__name__)
    store = SessionStore()
    search_engine = SemanticSearchEngine()
    try:
        search_engine.warmup()
    except Exception:
        pass

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/api/process")
    def api_process():
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided."}), 400

        uploaded = request.files["audio"]
        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Invalid audio file."}), 400

        audio_path = _save_upload(uploaded)
        error = _validate_audio(audio_path)
        if error:
            return jsonify({"error": error}), 400

        model_size = request.form.get("model_size", "small")
        language = "auto"
        enable_diarization = True
        enable_noise_reduction = request.form.get("noise_reduction", "true").lower() == "true"

        result = process_audio(
            audio_path=str(audio_path),
            model_size=model_size,
            language=language,
            enable_diarization=enable_diarization,
            enable_noise_reduction=enable_noise_reduction,
        )

        if result.get("error"):
            return jsonify({"error": result["error"]}), 500

        tts_path = result.get("tts_audio_path") or ""
        if tts_path:
            result["tts_audio_url"] = f"/api/tts/{Path(tts_path).name}"
        if result.get("metadata", {}).get("session_id"):
            result["audio_url"] = f"/api/audio/{result['metadata']['session_id']}"
        return jsonify(result)

    @app.get("/api/sessions")
    def api_sessions():
        return jsonify({"sessions": store.list_sessions()})

    @app.get("/api/speakers")
    def api_speakers():
        store_vp = VoiceprintStore()
        return jsonify({"speakers": store_vp.list_speakers_meta()})

    @app.post("/api/speakers/rename")
    def api_speakers_rename():
        payload = request.get_json(silent=True) or {}
        old_name = (payload.get("old_name") or "").strip()
        new_name = (payload.get("new_name") or "").strip()
        if not old_name or not new_name:
            return jsonify({"error": "Missing old_name or new_name."}), 400
        store_vp = VoiceprintStore()
        if not store_vp.rename_speaker(old_name, new_name):
            return jsonify({"error": "Rename failed. Check names."}), 400
        return jsonify({"status": "renamed", "old_name": old_name, "new_name": new_name})

    @app.post("/api/speakers/delete")
    def api_speakers_delete():
        payload = request.get_json(silent=True) or {}
        name = (payload.get("name") or "").strip()
        if not name:
            return jsonify({"error": "Missing name."}), 400
        store_vp = VoiceprintStore()
        if not store_vp.delete_speaker(name):
            return jsonify({"error": "Delete failed. Speaker not found."}), 404
        return jsonify({"status": "deleted", "name": name})

    @app.get("/api/status")
    def api_status():
        voiceprint_model = EMBEDDINGS_DIR / "voiceprint.onnx"
        sbert_model = EMBEDDINGS_DIR / "sentence_bert.onnx"
        sbert_tokenizer = EMBEDDINGS_DIR / "sentence_bert_tokenizer"
        try:
            import speechbrain  # noqa: F401
            voice_torch = True
        except Exception:
            voice_torch = False
        try:
            import transformers  # noqa: F401
            semantic_torch = True
        except Exception:
            semantic_torch = False
        try:
            import onnxruntime  # noqa: F401
            onnx_ready = True
        except Exception:
            onnx_ready = False
        try:
            import sklearn  # noqa: F401
            sklearn_ready = True
        except Exception:
            sklearn_ready = False
        voice_ready = voiceprint_model.exists() or voice_torch
        semantic_ready = (sbert_model.exists() and sbert_tokenizer.exists()) or semantic_torch
        return jsonify({
            "voiceprint_model": voiceprint_model.exists(),
            "semantic_model": sbert_model.exists(),
            "semantic_tokenizer": sbert_tokenizer.exists(),
            "voice_torch": voice_torch,
            "semantic_torch": semantic_torch,
            "voice_ready": voice_ready,
            "semantic_ready": semantic_ready,
            "onnxruntime": onnx_ready,
            "sklearn": sklearn_ready,
            "embeddings_dir": str(EMBEDDINGS_DIR),
        })

    @app.get("/api/sessions/<session_id>")
    def api_session(session_id: str):
        data = store.load_session(session_id)
        if not data:
            return jsonify({"error": "Session not found."}), 404
        if data.get("audio_file"):
            data["audio_url"] = f"/api/audio/{session_id}"
        if data.get("tts_audio"):
            data["tts_audio_url"] = f"/api/tts/{data['tts_audio']}"
        return jsonify(data)

    @app.post("/api/sessions/<session_id>/delete")
    def api_delete_session(session_id: str):
        if not store.delete_session(session_id):
            return jsonify({"error": "Session not found."}), 404
        return jsonify({"status": "deleted", "session_id": session_id})

    @app.get("/api/audio/<session_id>")
    def api_audio(session_id: str):
        session_dir = SESSIONS_DIR / session_id
        if not session_dir.exists():
            return jsonify({"error": "Session not found."}), 404
        metadata_path = session_dir / "metadata.json"
        if not metadata_path.exists():
            return jsonify({"error": "Metadata not found."}), 404
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        audio_file = metadata.get("audio_file")
        if not audio_file:
            return jsonify({"error": "Audio not found."}), 404
        audio_path = session_dir / audio_file
        if not audio_path.exists():
            return jsonify({"error": "Audio not found."}), 404
        mime = mimetypes.guess_type(str(audio_path))[0] or "audio/wav"
        return send_file(audio_path, mimetype=mime, as_attachment=False)

    @app.post("/api/clear")
    def api_clear():
        store.clear_all()
        for folder in (OUTPUT_DIR, TEMP_DIR, DATA_DIR):
            if folder.exists():
                for path in folder.iterdir():
                    if path.is_file():
                        path.unlink()
        return jsonify({"status": "cleared"})

    @app.post("/api/bookmarks")
    def api_bookmarks():
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
        if not session_id:
            return jsonify({"error": "Missing session_id."}), 400
        bookmark = {
            "timestamp": float(payload.get("timestamp", 0.0)),
            "snippet": payload.get("snippet") or "",
            "tags": payload.get("tags") or [],
            "created_at": payload.get("created_at") or "",
        }
        bookmarks = store.save_bookmark(session_id, bookmark)
        return jsonify({"bookmarks": bookmarks})

    @app.post("/api/search")
    def api_search():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing query."}), 400
        mode = payload.get("mode", "smart")
        session_id = payload.get("session_id")
        session_ids = [session_id] if session_id else None

        results: List[Dict[str, Any]] = []
        if mode == "smart":
            results = search_engine.search(query, session_ids=session_ids)
            if not results:
                results = search_engine.keyword_search(query, session_ids=session_ids)
        else:
            results = search_engine.keyword_search(query, session_ids=session_ids)
        return jsonify({"results": results})

    def _find_ffmpeg() -> Optional[str]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg
        base = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
        pattern = str(base / "Gyan.FFmpeg*" / "ffmpeg" / "bin" / "ffmpeg.exe")
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
        return None

    def _load_audio_mono(path: Path, target_sr: int = 16000):
        from scipy.signal import resample_poly
        last_exc: Optional[Exception] = None
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), always_2d=True)
            audio = data.mean(axis=1)
        except Exception as exc:
            last_exc = exc
            ffmpeg_first = path.suffix.lower() in {".m4a", ".mp4", ".aac", ".m4b"}
            ffmpeg = _find_ffmpeg() if ffmpeg_first else None
            if ffmpeg:
                try:
                    os.environ["PATH"] = f"{Path(ffmpeg).parent};{os.environ.get('PATH','')}"
                    from pydub import AudioSegment
                    AudioSegment.converter = ffmpeg
                    AudioSegment.ffmpeg = ffmpeg
                    AudioSegment.ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
                    audio_seg = AudioSegment.from_file(str(path))
                    audio_seg = audio_seg.set_channels(1).set_frame_rate(target_sr)
                    samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
                    audio = samples / (1 << (8 * audio_seg.sample_width - 1))
                    sr = target_sr
                except Exception as exc:
                    last_exc = exc
                else:
                    if sr != target_sr:
                        audio = resample_poly(audio, target_sr, sr)
                        sr = target_sr
                    return audio.astype(np.float32), sr

            try:
                import av
                try:
                    container = av.open(str(path))
                except Exception as exc:
                    raise RuntimeError(f"Unsupported format {path.suffix}. Install ffmpeg or use WAV/FLAC/OGG.") from exc
                stream = container.streams.audio[0]
                resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_sr)
                frames = []
                for frame in container.decode(stream):
                    frame = resampler.resample(frame)
                    if frame is None:
                        continue
                    frames.append(frame.to_ndarray())
                container.close()
                if not frames:
                    raise RuntimeError(f"No audio frames decoded for {path.suffix}.")
                audio = np.concatenate(frames, axis=1).flatten().astype(np.float32)
                audio = audio / 32768.0
                sr = target_sr
            except Exception as exc:
                last_exc = exc
                ffmpeg = _find_ffmpeg()
                if not ffmpeg:
                    detail = f" ({last_exc})" if last_exc else ""
                    raise RuntimeError(f"Unsupported format {path.suffix}. Install ffmpeg or use WAV/FLAC/OGG.{detail}")
                os.environ["PATH"] = f"{Path(ffmpeg).parent};{os.environ.get('PATH','')}"
                from pydub import AudioSegment
                AudioSegment.converter = ffmpeg
                AudioSegment.ffmpeg = ffmpeg
                AudioSegment.ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe")
                audio_seg = AudioSegment.from_file(str(path))
                audio_seg = audio_seg.set_channels(1).set_frame_rate(target_sr)
                samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
                audio = samples / (1 << (8 * audio_seg.sample_width - 1))
                sr = target_sr

        if sr != target_sr:
            audio = resample_poly(audio, target_sr, sr)
            sr = target_sr
        return audio.astype(np.float32), sr

    def _safe_unlink(path: Path) -> None:
        for _ in range(3):
            try:
                path.unlink(missing_ok=True)
                return
            except PermissionError:
                time.sleep(0.2)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            return

    @app.post("/api/enroll")
    def api_enroll():
        speaker_name = request.form.get("speaker_name", "").strip()
        if not speaker_name:
            return jsonify({"error": "Missing speaker_name."}), 400
        files = request.files.getlist("samples")
        if not files:
            return jsonify({"error": "Provide 3-5 audio samples."}), 400

        embedder = VoiceprintEmbedder()
        if not embedder.available():
            return jsonify({"error": "Voiceprint model not found. Place voiceprint.onnx in ~/.smartvoice/embeddings."}), 400

        embeddings = []
        accepted = 0
        rejected = 0
        durations = []
        errors = []
        for sample in files[:5]:
            suffix = Path(sample.filename).suffix.lower() or ".wav"
            temp_path = TEMP_DIR / f"enroll_{uuid.uuid4().hex[:6]}{suffix}"
            sample.save(temp_path)
            try:
                y, sr = _load_audio_mono(temp_path, target_sr=16000)
            except Exception as exc:
                errors.append(f"{sample.filename}: {exc}")
                _safe_unlink(temp_path)
                rejected += 1
                continue
            duration = len(y) / float(sr or 16000)
            durations.append(duration)
            if duration < 5 or duration > 60:
                _safe_unlink(temp_path)
                rejected += 1
                errors.append(f"{sample.filename}: {duration:.1f}s")
                continue
            embeddings.append(embedder.embed_audio(y))
            accepted += 1
            _safe_unlink(temp_path)

        if len(embeddings) < 3:
            msg = "Need at least 3 valid samples (5-60s each)."
            if durations:
                msg += f" Accepted {accepted}, rejected {rejected}."
            if errors:
                msg += " Issues: " + "; ".join(errors[:3])
            return jsonify({"error": msg}), 400

        store_vp = VoiceprintStore()
        data = store_vp.add_samples(speaker_name, embeddings)
        return jsonify({"status": "enrolled", "speaker": speaker_name, "samples": data.get("samples", 0)})

    @app.post("/api/export/<session_id>")
    def api_export(session_id: str):
        zip_path = store.export_session(session_id)
        if not zip_path:
            return jsonify({"error": "Session not found."}), 404
        return send_file(zip_path, mimetype="application/zip", as_attachment=True)

    @app.get("/api/tts/<name>")
    def get_tts(name: str):
        candidate = OUTPUT_DIR / name
        if not candidate.exists():
            return jsonify({"error": "Audio not found."}), 404
        return send_file(candidate, mimetype="audio/mpeg", as_attachment=False)

    return app


def main():
    """Launch the application."""
    app = create_app()
    app.run(host="127.0.0.1", port=7860, debug=False)


if __name__ == '__main__':
    main()
