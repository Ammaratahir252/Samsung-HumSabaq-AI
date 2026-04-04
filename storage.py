"""
SmartVoice - Local storage utilities
Offline-first persistence for sessions, embeddings, and bookmarks.
"""
from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional


BASE_STORE_DIR = Path.home() / ".smartvoice"
SESSIONS_DIR = BASE_STORE_DIR / "sessions"
EMBEDDINGS_DIR = BASE_STORE_DIR / "embeddings"
BOOKMARKS_DIR = BASE_STORE_DIR / "bookmarks"
EXPORTS_DIR = BASE_STORE_DIR / "exports"


def ensure_dirs() -> None:
    for path in (SESSIONS_DIR, EMBEDDINGS_DIR, BOOKMARKS_DIR, EXPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class SessionStore:
    def __init__(self) -> None:
        ensure_dirs()

    def session_dir(self, session_id: str) -> Path:
        return SESSIONS_DIR / session_id

    def save_session(self, state: Dict[str, Any], audio_path: Optional[str], formatted_transcript: str) -> Dict[str, Any]:
        ensure_dirs()
        session_id = state["session_id"]
        session_dir = self.session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)

        audio_target = None
        if audio_path:
            src = Path(audio_path)
            if src.exists():
                audio_target = session_dir / f"audio{src.suffix.lower()}"
                shutil.copy2(src, audio_target)

        transcript_path = session_dir / "transcript.txt"
        transcript_path.write_text(formatted_transcript, encoding="utf-8")

        segments_path = session_dir / "segments.json"
        _write_json(segments_path, state.get("segments", []))

        summary_path = session_dir / "summary.json"
        summary_payload = {
            "bullets": state.get("bullet_points", []),
            "keywords": state.get("keywords", []),
        }
        _write_json(summary_path, summary_payload)

        segments = state.get("segments", [])
        duration = 0.0
        if segments:
            duration = max(seg.get("end", 0.0) for seg in segments)

        metadata = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "language": state.get("detected_language"),
            "speakers": len(state.get("speaker_info", {})) or 1,
            "segments": len(segments),
            "duration": round(duration, 2),
            "bullets": len(state.get("bullet_points", [])),
            "audio_file": audio_target.name if audio_target else None,
            "tts_audio": Path(state.get("tts_audio_path") or "").name if state.get("tts_audio_path") else None,
        }
        _write_json(session_dir / "metadata.json", metadata)
        return metadata

    def list_sessions(self) -> List[Dict[str, Any]]:
        ensure_dirs()
        items: List[Dict[str, Any]] = []
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            meta = _read_json(session_dir / "metadata.json", None)
            if meta:
                items.append(meta)
        items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return items

    def load_session(self, session_id: str) -> Dict[str, Any]:
        session_dir = self.session_dir(session_id)
        if not session_dir.exists():
            return {}

        metadata = _read_json(session_dir / "metadata.json", {})
        transcript = (session_dir / "transcript.txt").read_text(encoding="utf-8") if (session_dir / "transcript.txt").exists() else ""
        summary = _read_json(session_dir / "summary.json", {})
        segments = _read_json(session_dir / "segments.json", [])
        bookmarks = self.list_bookmarks(session_id)

        return {
            "metadata": metadata,
            "transcript": transcript,
            "summary": summary,
            "segments": segments,
            "bookmarks": bookmarks,
            "audio_file": metadata.get("audio_file"),
            "tts_audio": metadata.get("tts_audio"),
        }

    def save_bookmark(self, session_id: str, bookmark: Dict[str, Any]) -> List[Dict[str, Any]]:
        ensure_dirs()
        path = BOOKMARKS_DIR / f"{session_id}.json"
        payload = _read_json(path, [])
        payload.append(bookmark)
        _write_json(path, payload)
        return payload

    def list_bookmarks(self, session_id: str) -> List[Dict[str, Any]]:
        path = BOOKMARKS_DIR / f"{session_id}.json"
        return _read_json(path, [])

    def clear_all(self) -> None:
        if BASE_STORE_DIR.exists():
            shutil.rmtree(BASE_STORE_DIR, ignore_errors=True)
        ensure_dirs()

    def delete_session(self, session_id: str) -> bool:
        session_dir = self.session_dir(session_id)
        if not session_dir.exists():
            return False
        shutil.rmtree(session_dir, ignore_errors=True)
        bookmark_path = BOOKMARKS_DIR / f"{session_id}.json"
        if bookmark_path.exists():
            bookmark_path.unlink()
        return True

    def export_session(self, session_id: str) -> Optional[Path]:
        session_dir = self.session_dir(session_id)
        if not session_dir.exists():
            return None
        ensure_dirs()
        zip_path = EXPORTS_DIR / f"smartvoice_{session_id}.zip"
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in session_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=f"{session_id}/{path.relative_to(session_dir)}")
            bookmark_path = BOOKMARKS_DIR / f"{session_id}.json"
            if bookmark_path.exists():
                zf.write(bookmark_path, arcname=f"{session_id}/bookmarks.json")
        return zip_path
