
from __future__ import annotations
import os, re, json, uuid, logging, importlib, types
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import TypedDict, Annotated, Literal, Optional, List, Dict, Any
from functools import lru_cache
from collections import Counter

import numpy as np

import requests

# Ensure SpeechBrain does not try to load optional k2 on Windows.
os.environ.setdefault("SPEECHBRAIN_DISABLE_K2", "1")
os.environ.setdefault("SB_DISABLE_K2", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

BASE_DIR = Path(__file__).parent

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=BASE_DIR / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
TEMP_DIR = BASE_DIR / 'temp'
for p in (DATA_DIR, OUTPUT_DIR, TEMP_DIR): p.mkdir(parents=True, exist_ok=True)

ALLOWED_AUDIO = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
MAX_AUDIO_MB = 250
HF_TOKEN = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN', '')

STOPWORDS = frozenset({
    'the','is','are','a','an','and','or','to','of','in','for','on','with','that',
    'this','it','as','be','by','from','at','was','were','has','have','had','we',
    'you','they','he','she','i','but','if','then','so','what','when','where','who',
    'um','uh','okay','ok','yeah','yes','right','actually','basically','really'
})

try:
    from HamSabaqAi.SmartVoice.storage import EMBEDDINGS_DIR, SESSIONS_DIR, ensure_dirs
except ModuleNotFoundError:
    from SmartVoice.storage import EMBEDDINGS_DIR, SESSIONS_DIR, ensure_dirs

# ==================== VOICEPRINTS (LOCAL) ====================
DEFAULT_VOICEPRINT_MODEL = "voiceprint.onnx"
DEFAULT_VOICEPRINT_TORCH = "speechbrain/spkrec-ecapa-voxceleb"
VOICEPRINTS_FILE = EMBEDDINGS_DIR / "voiceprints.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VoiceprintEmbedder:
    def __init__(self, model_path: Optional[Path] = None) -> None:
        ensure_dirs()
        self.model_path = model_path or (EMBEDDINGS_DIR / DEFAULT_VOICEPRINT_MODEL)
        self._session = None
        self._torch_model = None

    def available(self) -> bool:
        if self.model_path.exists():
            return True
        try:
            import speechbrain  # noqa: F401
            return True
        except Exception:
            return False

    def _get_session(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        return self._session

    def _get_torch_model(self):
        if self._torch_model is not None:
            return self._torch_model
        os.environ.setdefault("SPEECHBRAIN_DISABLE_K2", "1")
        os.environ.setdefault("SB_DISABLE_K2", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            import torch
            if not hasattr(torch, "amp"):
                torch.amp = types.SimpleNamespace()
            if not hasattr(torch.amp, "custom_fwd"):
                from torch.cuda.amp import custom_fwd as cuda_custom_fwd, custom_bwd as cuda_custom_bwd

                def _custom_fwd(*args, **kwargs):
                    kwargs.pop("device_type", None)
                    return cuda_custom_fwd(*args, **kwargs)

                def _custom_bwd(*args, **kwargs):
                    kwargs.pop("device_type", None)
                    return cuda_custom_bwd(*args, **kwargs)

                torch.amp.custom_fwd = _custom_fwd
                torch.amp.custom_bwd = _custom_bwd
        except Exception:
            pass
        from speechbrain.inference import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        self._torch_model = EncoderClassifier.from_hparams(
            source=DEFAULT_VOICEPRINT_TORCH,
            savedir=str(EMBEDDINGS_DIR / "ecapa_model"),
            local_strategy=LocalStrategy.COPY,
        )
        self._torch_model.eval()
        return self._torch_model

    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        if self.model_path.exists():
            session = self._get_session()
            input_name = session.get_inputs()[0].name
            if audio.ndim == 1:
                audio = audio[None, :]
            audio = audio.astype(np.float32)
            outputs = session.run(None, {input_name: audio})
            embedding = outputs[0][0]
            return np.array(embedding, dtype=np.float32)

        import torch
        model = self._get_torch_model()
        if audio.ndim == 1:
            audio = audio[None, :]
        wav = torch.from_numpy(audio.astype(np.float32))
        with torch.no_grad():
            emb = model.encode_batch(wav)
        return emb.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)


class VoiceprintStore:
    def __init__(self) -> None:
        ensure_dirs()
        self._data = _read_json(VOICEPRINTS_FILE, {"speakers": {}})

    def _reload(self) -> None:
        self._data = _read_json(VOICEPRINTS_FILE, {"speakers": {}})

    def list_speakers(self) -> List[str]:
        self._reload()
        return sorted(self._data.get("speakers", {}).keys())

    def list_speakers_meta(self) -> List[Dict[str, Any]]:
        self._reload()
        speakers = []
        for name, data in self._data.get("speakers", {}).items():
            speakers.append({
                "name": name,
                "samples": data.get("samples", 0),
                "updated_at": data.get("updated_at", ""),
            })
        speakers.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return speakers

    def add_samples(self, speaker: str, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        if not embeddings:
            return {}
        self._reload()
        speaker_data = self._data.setdefault("speakers", {}).setdefault(speaker, {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "embeddings": [],
            "mean": [],
            "samples": 0,
        })
        for emb in embeddings:
            speaker_data["embeddings"].append(emb.tolist())
        speaker_data["samples"] = len(speaker_data["embeddings"])
        speaker_data["updated_at"] = datetime.now().isoformat()
        mean = np.mean(np.array(speaker_data["embeddings"], dtype=np.float32), axis=0)
        speaker_data["mean"] = mean.tolist()
        speaker_data["threshold"] = self._calibrate_threshold(speaker_data["embeddings"], mean)
        _write_json(VOICEPRINTS_FILE, self._data)
        return speaker_data

    def get_threshold(self, speaker: Optional[str]) -> Optional[float]:
        if not speaker:
            return None
        self._reload()
        payload = self._data.get("speakers", {}).get(speaker, {})
        value = payload.get("threshold")
        return float(value) if value is not None else None

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        self._reload()
        best_name = None
        best_score = 0.0
        for name, payload in self._data.get("speakers", {}).items():
            mean = np.array(payload.get("mean", []), dtype=np.float32)
            if mean.size == 0:
                continue
            score = _cosine_similarity(embedding, mean)
            if score > best_score:
                best_name = name
                best_score = score
        return best_name, best_score

    def rename_speaker(self, old_name: str, new_name: str) -> bool:
        old_name = (old_name or "").strip()
        new_name = (new_name or "").strip()
        if not old_name or not new_name or old_name == new_name:
            return False
        self._reload()
        speakers = self._data.get("speakers", {})
        if old_name not in speakers or new_name in speakers:
            return False
        speakers[new_name] = speakers.pop(old_name)
        speakers[new_name]["updated_at"] = datetime.now().isoformat()
        _write_json(VOICEPRINTS_FILE, self._data)
        return True

    def delete_speaker(self, name: str) -> bool:
        name = (name or "").strip()
        if not name:
            return False
        self._reload()
        speakers = self._data.get("speakers", {})
        if name not in speakers:
            return False
        speakers.pop(name, None)
        _write_json(VOICEPRINTS_FILE, self._data)
        return True

    def _calibrate_threshold(self, embeddings: List[List[float]], mean: np.ndarray) -> float:
        if not embeddings:
            return 0.75
        vectors = np.array(embeddings, dtype=np.float32)
        sims = []
        for vec in vectors:
            sims.append(_cosine_similarity(vec, mean))
        mean_sim = float(np.mean(sims)) if sims else 0.75
        std_sim = float(np.std(sims)) if sims else 0.0
        return max(0.6, mean_sim - std_sim)


class SpeakerResolver:
    def __init__(self, confidence_threshold: float = 0.4, continuity_gap: float = 1.5) -> None:
        self.confidence_threshold = confidence_threshold
        self.continuity_gap = continuity_gap
        self.embedder = VoiceprintEmbedder()
        self.store = VoiceprintStore()

    def available(self) -> bool:
        return self.embedder.available() and bool(self.store.list_speakers())

    def resolve(self, segments: List[Dict[str, Any]], audio_provider) -> List[Dict[str, Any]]:
        return self.resolve_with_cache(segments, audio_provider, cache_path=None)

    def resolve_with_cache(self, segments: List[Dict[str, Any]], audio_provider, cache_path: Optional[Path]) -> List[Dict[str, Any]]:
        if not segments:
            return segments
        if not self.embedder.available():
            return segments

        clusters: List[Dict[str, Any]] = []
        cached: List[Dict[str, Any]] = []

        cached_embeddings = self._load_embedding_cache(cache_path, len(segments)) if cache_path else None

        for idx, seg in enumerate(segments):
            if cached_embeddings is not None:
                emb = cached_embeddings[idx]
                if emb is None:
                    cached.append({"name": None, "score": 0.0, "embedding": None})
                    continue
                name, score = self.store.match(emb)
                cached.append({"name": name, "score": score, "embedding": emb})
                continue

            audio = audio_provider(seg["start"], seg["end"])
            if audio is None or audio.size == 0:
                cached.append({"name": None, "score": 0.0, "embedding": None})
                continue
            emb = self.embedder.embed_audio(audio)
            name, score = self.store.match(emb)
            cached.append({"name": name, "score": score, "embedding": emb})

        if cache_path and cached_embeddings is None:
            self._save_embedding_cache(cache_path, cached)

        cluster_labels: List[int] = []
        cluster_centroids: Dict[int, np.ndarray] = {}
        cluster_speaker: Dict[int, Optional[str]] = {}
        cluster_score: Dict[int, float] = {}
        try:
            cluster_labels, cluster_centroids = self._cluster_all([c["embedding"] for c in cached if c["embedding"] is not None])
            if cluster_labels:
                cluster_speaker, cluster_score = self._map_clusters_to_speakers(cluster_centroids)
        except Exception:
            cluster_labels = []

        last_label = None
        last_end = None
        last_conf = 0.0

        cluster_cursor = 0
        for idx, seg in enumerate(segments):
            cached_item = cached[idx]
            label = cached_item["name"]
            confidence = cached_item["score"]

            cluster_id = None
            if cached_item["embedding"] is not None and cluster_labels:
                cluster_id = cluster_labels[cluster_cursor]
                cluster_cursor += 1

            if label is None and cluster_id is not None:
                clustered_name = cluster_speaker.get(cluster_id)
                clustered_score = cluster_score.get(cluster_id, 0.0)
                if clustered_name:
                    label = clustered_name
                    confidence = clustered_score

            if label is not None:
                threshold = self.store.get_threshold(label) or 0.35
                if confidence < threshold:
                    label = label

            if label is None:
                if cluster_id is not None:
                    label = f"Speaker {cluster_id + 1}"
                    confidence = max(confidence, cluster_score.get(cluster_id, 0.0))
                elif cached_item["embedding"] is not None:
                    cluster_id = self._assign_cluster(cached_item["embedding"], clusters)
                    label = f"Speaker {cluster_id + 1}"
                    confidence = clusters[cluster_id]["score"]
                else:
                    label = "Unknown Speaker"
                    confidence = 0.0

            if last_label and last_end is not None and seg["start"] - last_end <= self.continuity_gap:
                if confidence < self.confidence_threshold and last_conf >= confidence:
                    label = last_label
                    confidence = max(confidence, last_conf)

            if confidence < self.confidence_threshold and not str(label).endswith("?"):
                label = f"{label} ?"

            seg["speaker"] = label
            seg["speaker_confidence"] = round(confidence, 3)
            last_label = label
            last_end = seg["end"]
            last_conf = confidence

        self._smooth_labels(segments)
        return segments

    def _cluster_all(self, embeddings: List[np.ndarray], distance_threshold: float = 0.35) -> Tuple[List[int], Dict[int, np.ndarray]]:
        if not embeddings:
            return [], {}
        try:
            from sklearn.cluster import AgglomerativeClustering
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for agglomerative clustering") from exc

        matrix = np.stack(embeddings, axis=0)
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                metric="cosine",
                linkage="average",
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                affinity="cosine",
                linkage="average",
            )
        labels = clustering.fit_predict(matrix)

        centroids: Dict[int, np.ndarray] = {}
        for label in np.unique(labels):
            centroids[int(label)] = np.mean(matrix[labels == label], axis=0)

        return labels.tolist(), centroids

    def _map_clusters_to_speakers(self, centroids: Dict[int, np.ndarray]) -> Tuple[Dict[int, Optional[str]], Dict[int, float]]:
        cluster_speaker: Dict[int, Optional[str]] = {}
        cluster_score: Dict[int, float] = {}
        for cluster_id, centroid in centroids.items():
            name, score = self.store.match(centroid)
            cluster_speaker[cluster_id] = name
            cluster_score[cluster_id] = score
        return cluster_speaker, cluster_score

    def _smooth_labels(self, segments: List[Dict[str, Any]]) -> None:
        if len(segments) < 3:
            return
        labels = [seg.get("speaker") for seg in segments]
        for idx in range(1, len(segments) - 1):
            prev_label = labels[idx - 1]
            next_label = labels[idx + 1]
            if prev_label == next_label and labels[idx] != prev_label:
                segments[idx]["speaker"] = prev_label
                prev_conf = segments[idx - 1].get("speaker_confidence", 0.0)
                next_conf = segments[idx + 1].get("speaker_confidence", 0.0)
                segments[idx]["speaker_confidence"] = round(max(prev_conf, next_conf), 3)

    def _load_embedding_cache(self, cache_path: Optional[Path], expected_len: int) -> Optional[List[Optional[np.ndarray]]]:
        if not cache_path or not cache_path.exists():
            return None
        try:
            payload = np.load(cache_path, allow_pickle=False)
            embeddings = payload["embeddings"]
            valid = payload["valid"] if "valid" in payload.files else None
            if embeddings.shape[0] != expected_len:
                return None
            if valid is None or valid.shape[0] != expected_len:
                return None
            items: List[Optional[np.ndarray]] = []
            for idx in range(embeddings.shape[0]):
                if bool(valid[idx]):
                    items.append(embeddings[idx])
                else:
                    items.append(None)
            return items
        except Exception:
            return None

    def _save_embedding_cache(self, cache_path: Path, cached: List[Dict[str, Any]]) -> None:
        try:
            vectors: List[np.ndarray] = []
            valid: List[bool] = []
            embed_size = None
            for item in cached:
                emb = item["embedding"]
                if emb is None:
                    valid.append(False)
                else:
                    valid.append(True)
                    embed_size = emb.shape[0]
            if embed_size is None:
                return
            for item in cached:
                emb = item["embedding"]
                if emb is None:
                    vectors.append(np.zeros((embed_size,), dtype=np.float32))
                else:
                    vectors.append(emb.astype(np.float32))
            np.savez_compressed(cache_path, embeddings=np.stack(vectors, axis=0), valid=np.array(valid))
        except Exception:
            return

    def _assign_cluster(self, emb: np.ndarray, clusters: List[Dict[str, Any]], threshold: float = 0.7) -> int:
        best_idx = None
        best_score = 0.0
        for idx, cluster in enumerate(clusters):
            denom = (np.linalg.norm(cluster["mean"]) * np.linalg.norm(emb)) + 1e-8
            score = float(np.dot(cluster["mean"], emb) / denom)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None or best_score < threshold:
            clusters.append({"mean": emb, "score": best_score})
            return len(clusters) - 1
        cluster = clusters[best_idx]
        cluster["mean"] = (cluster["mean"] + emb) / 2.0
        cluster["score"] = best_score
        return best_idx


# ==================== SEMANTIC SEARCH (LOCAL) ====================
DEFAULT_SEMANTIC_MODEL = "sentence_bert.onnx"
DEFAULT_TOKENIZER_DIR = "sentence_bert_tokenizer"
DEFAULT_SEMANTIC_TORCH = "sentence-transformers/all-MiniLM-L6-v2"


def _cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)


class SentenceEmbedder:
    def __init__(self, model_path: Optional[Path] = None, tokenizer_path: Optional[Path] = None) -> None:
        ensure_dirs()
        self.model_path = model_path or (EMBEDDINGS_DIR / DEFAULT_SEMANTIC_MODEL)
        self.tokenizer_path = tokenizer_path or (EMBEDDINGS_DIR / DEFAULT_TOKENIZER_DIR)
        self._session = None
        self._tokenizer = None
        self._torch_model = None

    def available(self) -> bool:
        if self.model_path.exists() and self.tokenizer_path.exists():
            return True
        try:
            import transformers  # noqa: F401
            return True
        except Exception:
            return False

    def _get_session(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)
        return self._session

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from transformers import AutoTokenizer
        if self.model_path.exists() and self.tokenizer_path.exists():
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path), local_files_only=True)
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SEMANTIC_TORCH)
        return self._tokenizer

    def _get_torch_model(self):
        if self._torch_model is not None:
            return self._torch_model
        from transformers import AutoModel
        self._torch_model = AutoModel.from_pretrained(DEFAULT_SEMANTIC_TORCH)
        self._torch_model.eval()
        return self._torch_model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        tokenizer = self._get_tokenizer()
        if self.model_path.exists() and self.tokenizer_path.exists():
            session = self._get_session()
            encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
            inputs = {name: encoded[name] for name in encoded}
            outputs = session.run(None, inputs)
            embeddings = outputs[0]
            if embeddings.ndim == 3:
                embeddings = embeddings.mean(axis=1)
            return embeddings.astype(np.float32)

        import torch
        model = self._get_torch_model()
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output = model(**encoded)
        last_hidden = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = summed / counts
        return pooled.cpu().numpy().astype(np.float32)


class SemanticSearchEngine:
    def __init__(self) -> None:
        ensure_dirs()
        self.embedder = SentenceEmbedder()

    def warmup(self) -> None:
        if not self.embedder.available():
            return
        for session_dir in SESSIONS_DIR.iterdir():
            if not session_dir.is_dir():
                continue
            session_id = session_dir.name
            segments = _read_json(session_dir / "segments.json", [])
            if not segments:
                continue
            index_path = EMBEDDINGS_DIR / f"semantic_{session_id}.npz"
            if index_path.exists():
                continue
            self.index_session(session_id, segments)

    def index_session(self, session_id: str, segments: List[Dict[str, Any]]) -> Optional[Path]:
        if not self.embedder.available():
            return None
        texts = [seg.get("text", "") for seg in segments]
        embeddings = self.embedder.embed(texts)
        index_path = EMBEDDINGS_DIR / f"semantic_{session_id}.npz"
        np.savez_compressed(index_path, embeddings=embeddings)
        _write_json(EMBEDDINGS_DIR / f"semantic_{session_id}.json", {
            "session_id": session_id,
            "count": len(texts),
        })
        return index_path

    def _load_index(self, session_id: str) -> Optional[np.ndarray]:
        index_path = EMBEDDINGS_DIR / f"semantic_{session_id}.npz"
        if not index_path.exists():
            return None
        return np.load(index_path)["embeddings"]

    def search(self, query: str, session_ids: Optional[List[str]], top_k: int = 5, min_score: float = 0.45) -> List[Dict[str, Any]]:
        if not self.embedder.available():
            return []
        targets = session_ids or [p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()]
        query_emb = self.embedder.embed([query])
        results: List[Dict[str, Any]] = []

        for session_id in targets:
            segments_path = SESSIONS_DIR / session_id / "segments.json"
            segments = _read_json(segments_path, [])
            if not segments:
                continue
            embeddings = self._load_index(session_id)
            if embeddings is None or embeddings.shape[0] != len(segments):
                self.index_session(session_id, segments)
                embeddings = self._load_index(session_id)
            if embeddings is None:
                continue

            sims = _cosine_sim_matrix(embeddings, query_emb).flatten()
            top_idx = sims.argsort()[::-1][:top_k]
            for idx in top_idx:
                score = float(sims[idx])
                if score < min_score:
                    continue
                seg = segments[idx]
                context = self._context_window(segments, idx)
                results.append({
                    "session_id": session_id,
                    "score": round(score, 3),
                    "segment": seg,
                    "context": context,
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def keyword_search(self, query: str, session_ids: Optional[List[str]], top_k: int = 5) -> List[Dict[str, Any]]:
        targets = session_ids or [p.name for p in SESSIONS_DIR.iterdir() if p.is_dir()]
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()

        for session_id in targets:
            segments_path = SESSIONS_DIR / session_id / "segments.json"
            segments = _read_json(segments_path, [])
            for idx, seg in enumerate(segments):
                text = seg.get("text", "")
                if query_lower in text.lower():
                    results.append({
                        "session_id": session_id,
                        "score": 1.0,
                        "segment": seg,
                        "context": self._context_window(segments, idx),
                    })
                    if len(results) >= top_k:
                        return results
        return results

    def _context_window(self, segments: List[Dict[str, Any]], idx: int, radius: int = 2) -> List[Dict[str, Any]]:
        start = max(0, idx - radius)
        end = min(len(segments), idx + radius + 1)
        return segments[start:end]

# ==================== STATE SCHEMA ====================
class AgentState(TypedDict):
    """Shared state across all agents in the workflow."""
    # Input
    audio_path: str
    model_size: str
    language: str
    enable_diarization: bool
    enable_noise_reduction: bool
    
    # Processing outputs
    processed_audio_path: str
    speaker_timeline: Dict[float, str]
    speaker_info: Dict[str, Any]
    segments: List[Dict]
    full_transcript: str
    detected_language: str
    
    # Summary outputs
    bullet_points: List[str]
    keywords: List[str]
    speaker_summaries: Dict[str, List[str]]
    
    # Final outputs
    tts_audio_path: str
    session_id: str
    error: str
    status: str



# ==================== PROMPT TEMPLATES ====================
SUMMARIZATION_PROMPT = PromptTemplate.from_template("""
You are an expert meeting summarizer. Create concise bullet points from this transcript.

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
- Extract 3-5 key discussion points
- Each bullet should be a complete thought (10-20 words)
- Focus on decisions, action items, and important information
- Use third person ("The speaker discussed..." not "I discussed...")
- Be factual, avoid interpretation

OUTPUT FORMAT (JSON):
{{"bullets": ["point 1", "point 2", "point 3"]}}
""")

SPEAKER_SUMMARY_PROMPT = PromptTemplate.from_template("""
Summarize what {speaker} said in 1-2 sentences.

{speaker}'s statements:
{text}

Summary:
""")

KEYWORD_EXTRACTION_PROMPT = PromptTemplate.from_template("""
Extract 8-12 key topics/keywords from this transcript. Return as JSON list.

TRANSCRIPT:
{transcript}

OUTPUT: {{"keywords": ["keyword1", "keyword2", ...]}}
""")


# ==================== AUDIO PROCESSING AGENT ====================
class AudioProcessingAgent:
    """Agent for audio validation, preprocessing, and noise reduction."""
    
    def __init__(self):
        self._imports_loaded = False
    
    def _load_imports(self):
        if self._imports_loaded: return
        global sf, nr, resample_poly
        import soundfile as sf
        import noisereduce as nr
        from scipy.signal import resample_poly
        self._imports_loaded = True

    def _load_audio_mono(self, path: Path, target_sr: int = 16000):
        try:
            data, sr = sf.read(str(path), always_2d=True)
            audio = data.mean(axis=1)
        except Exception:
            try:
                import av
                container = av.open(str(path))
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
                    raise RuntimeError("No audio frames decoded.")
                audio = np.concatenate(frames, axis=1).flatten().astype(np.float32)
                audio = audio / 32768.0
                sr = target_sr
            except Exception:
                from pydub import AudioSegment
                audio_seg = AudioSegment.from_file(str(path))
                audio_seg = audio_seg.set_channels(1)
                sr = int(audio_seg.frame_rate)
                samples = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
                audio = samples / (1 << (8 * audio_seg.sample_width - 1))

        if sr != target_sr:
            audio = resample_poly(audio, target_sr, sr)
            sr = target_sr
        return audio.astype(np.float32), sr
    
    def __call__(self, state: AgentState) -> AgentState:
        """Process audio: validate, convert, denoise."""
        try:
            self._load_imports()
            audio_path = Path(state['audio_path'])
            
            # Validate
            if not audio_path.exists():
                return {**state, 'error': f'File not found: {audio_path}', 'status': 'failed'}
            if audio_path.suffix.lower() not in ALLOWED_AUDIO:
                return {**state, 'error': f'Invalid format: {audio_path.suffix}', 'status': 'failed'}
            if audio_path.stat().st_size / (1024*1024) > MAX_AUDIO_MB:
                return {**state, 'error': f'File too large (max {MAX_AUDIO_MB}MB)', 'status': 'failed'}
            
            # Convert to mono 16kHz WAV
            temp_path = TEMP_DIR / f'{audio_path.stem}_{uuid.uuid4().hex[:6]}.wav'
            ext = audio_path.suffix.lower()
            try:
                y, sr = self._load_audio_mono(audio_path, target_sr=16000)
                sf.write(str(temp_path), y, sr)
            except Exception as e:
                return {
                    **state,
                    'error': f'Audio decode failed: {e}',
                    'status': 'failed'
                }
            
            # Noise reduction
            if state.get('enable_noise_reduction', True):
                y, sr = self._load_audio_mono(temp_path, target_sr=16000)
                y_clean = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.75)
                clean_path = TEMP_DIR / f'{audio_path.stem}_clean_{uuid.uuid4().hex[:6]}.wav'
                sf.write(str(clean_path), y_clean, sr)
                logger.info(f"Audio processed with noise reduction: {clean_path}")
                return {**state, 'processed_audio_path': str(clean_path), 'status': 'audio_processed'}
            
            logger.info(f"Audio processed: {temp_path}")
            return {**state, 'processed_audio_path': str(temp_path), 'status': 'audio_processed'}
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {**state, 'error': str(e), 'status': 'failed'}


# ==================== DIARIZATION AGENT ====================
class DiarizationAgent:
    """Agent for speaker diarization using pyannote.audio."""
    
    def __init__(self):
        self._pipeline = None
        self._load_error = None
    
    def _get_pipeline(self):
        if self._pipeline:
            return self._pipeline
        if not HF_TOKEN:
            self._load_error = "HF_TOKEN missing"
            logger.warning("Diarization requires HF_TOKEN (Hugging Face access token)")
            return None
        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                'pyannote/speaker-diarization-3.0',
                use_auth_token=HF_TOKEN if HF_TOKEN else None
            )
            self._load_error = None
            return self._pipeline
        except Exception as e:
            self._load_error = str(e)
            logger.warning(f"Diarization unavailable: {e}")
            return None
    
    def __call__(self, state: AgentState) -> AgentState:
        """Identify speakers in audio."""
        if not state.get('enable_diarization', True):
            return {**state, 'speaker_timeline': {}, 'speaker_info': {}, 'status': 'diarization_skipped'}
        
        try:
            pipeline = self._get_pipeline()
            if not pipeline:
                detail = self._load_error or "Diarization unavailable"
                return {
                    **state,
                    'speaker_timeline': {},
                    'speaker_info': {},
                    'diarization_error': detail,
                    'status': 'diarization_skipped'
                }
            
            diarization = pipeline(state['processed_audio_path'])
            
            speaker_map, timeline, info = {}, {}, {}
            counter = 0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_map:
                    counter += 1
                    speaker_map[speaker] = f'Speaker {counter}'
                    info[f'Speaker {counter}'] = {'duration': 0, 'segments': 0}
                
                label = speaker_map[speaker]
                timeline[turn.start] = label
                info[label]['duration'] += turn.end - turn.start
                info[label]['segments'] += 1
            
            logger.info(f"Diarization complete: {len(info)} speakers")
            return {**state, 'speaker_timeline': timeline, 'speaker_info': info, 'status': 'diarization_complete'}
            
        except Exception as e:
            logger.warning(f"Diarization failed: {e}")
            return {**state, 'speaker_timeline': {}, 'speaker_info': {}, 'status': 'diarization_skipped'}


# ==================== TRANSCRIPTION AGENT ====================
class TranscriptionAgent:
    """Agent for speech-to-text using Faster Whisper."""
    
    @lru_cache(maxsize=2)
    def _get_model(self, model_size: str):
        try:
            whisper_module = importlib.import_module('faster_whisper')
            WhisperModel = getattr(whisper_module, 'WhisperModel')
        except Exception as e:
            raise ImportError("Missing dependency 'faster_whisper'. Install with: pip install faster-whisper") from e
        return WhisperModel(model_size, device='cpu', compute_type='int8')
    
    def _match_speaker(self, start: float, timeline: Dict[float, str]) -> str:
        if not timeline: return 'Speaker s'
        times = sorted(timeline.keys())
        speaker = 'Speaker'
        for t in times:
            if t <= start: speaker = timeline[t]
            else: break
        return speaker
    
    def __call__(self, state: AgentState) -> AgentState:
        """Transcribe audio to text with speaker labels."""
        try:
            model = self._get_model(state.get('model_size', 'small'))
            lang = state.get('language')
            lang = None if lang in [None, 'auto', ''] else lang
            
            segments_iter, info = model.transcribe(
                state['processed_audio_path'],
                language=lang,
                vad_filter=True,
                beam_size=1,
                condition_on_previous_text=False
            )
            
            timeline = state.get('speaker_timeline', {})
            segments = []
            
            for seg in segments_iter:
                text = seg.text.strip()
                if text:
                    segments.append({
                        'start': round(seg.start, 2),
                        'end': round(seg.end, 2),
                        'text': text,
                        'speaker': self._match_speaker(seg.start, timeline)
                    })
            
            full_text = ' '.join(s['text'] for s in segments)
            logger.info(f"Transcription complete: {len(segments)} segments, lang={info.language}")
            
            return {
                **state,
                'segments': segments,
                'full_transcript': full_text,
                'detected_language': info.language,
                'status': 'transcription_complete'
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {**state, 'error': str(e), 'status': 'failed'}


# ==================== SPEAKER ATTRIBUTION AGENT ====================
class SpeakerAttributionAgent:
    """Resolve speakers using enrolled voiceprints + D.O.D. rules."""

    def __init__(self):
        self._resolver = None
        self._imports_loaded = False

    def _load_imports(self):
        if self._imports_loaded:
            return
        global sf
        import soundfile as sf
        self._imports_loaded = True

    def _get_resolver(self):
        if self._resolver is not None:
            return self._resolver
        self._resolver = SpeakerResolver()
        self._embeddings_dir = EMBEDDINGS_DIR
        return self._resolver

    def __call__(self, state: AgentState) -> AgentState:
        try:
            self._load_imports()
            resolver = self._get_resolver()
            if not resolver.embedder.available():
                return {
                    **state,
                    'speaker_attribution_error': 'Voiceprint model unavailable (onnx or speechbrain).',
                    'status': 'speaker_attribution_skipped'
                }

            audio_path = state.get('processed_audio_path')
            if not audio_path:
                return {
                    **state,
                    'speaker_attribution_error': 'Missing processed audio path.',
                    'status': 'speaker_attribution_skipped'
                }

            try:
                y, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
                if y.ndim > 1:
                    y = y.mean(axis=1)
            except Exception as e:
                return {
                    **state,
                    'speaker_attribution_error': f'Failed to read audio for attribution: {e}',
                    'status': 'speaker_attribution_skipped'
                }

            def audio_provider(start: float, end: float):
                start_idx = max(0, int(float(start) * sr))
                end_idx = max(start_idx + 1, int(float(end) * sr))
                return y[start_idx:end_idx]

            session_id = state.get('session_id', 'session')
            cache_path = None
            if getattr(self, '_embeddings_dir', None):
                cache_path = self._embeddings_dir / f"speaker_{session_id}.npz"
            segments = resolver.resolve_with_cache(state.get('segments', []), audio_provider, cache_path)
            speaker_info: Dict[str, Any] = {}
            for seg in segments:
                label = seg.get('speaker', 'Speaker')
                speaker_info.setdefault(label, {'duration': 0, 'segments': 0})
                speaker_info[label]['duration'] += float(seg.get('end', 0)) - float(seg.get('start', 0))
                speaker_info[label]['segments'] += 1
            return {**state, 'segments': segments, 'speaker_info': speaker_info, 'status': 'speaker_attribution_complete'}
        except Exception as e:
            logger.warning(f"Speaker attribution failed: {e}")
            return {
                **state,
                'speaker_attribution_error': str(e),
                'status': 'speaker_attribution_skipped'
            }


# ==================== SUMMARIZATION AGENT ====================
class SummarizationAgent:
    """Agent for LLM-based abstractive summarization using Ollama (llama3.2:1b)."""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", model_name: str = "llama3.2:1b"):
        self.ollama_url = ollama_url.rstrip('/')
        self.model_name = model_name
    
    def _call_ollama(self, prompt: str, max_tokens: int = 300) -> Optional[str]:
        """Send a prompt to Ollama and return the generated text."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens, "temperature": 0.3}
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                logger.warning(f"Ollama error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.warning(f"Ollama request failed: {e}")
            return None
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        filtered = [t for t in tokens if t not in STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(top_n)]
    
    def _extractive_fallback(self, text: str, n_bullets: int = 4) -> List[str]:
        """Fallback extractive summarization (same as before)."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(sentences) <= n_bullets:
            return sentences
        
        freq = Counter(re.findall(r'\b[a-zA-Z]{3,}\b', text.lower()))
        scored = []
        for i, sent in enumerate(sentences):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
            if 5 <= len(sent.split()) <= 40:
                score = sum(freq.get(w, 0) for w in words)
                score += 3 if i < len(sentences) * 0.2 else 0
                scored.append((i, score, sent))
        
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:n_bullets]
        return [s[2] for s in sorted(top, key=lambda x: x[0])]
    
    def __call__(self, state: AgentState) -> AgentState:
        """Generate summary bullets and keywords using Ollama (or fallback)."""
        try:
            text = state.get('full_transcript', '')
            if not text.strip():
                return {**state, 'bullet_points': [], 'keywords': [], 'status': 'summarization_complete'}
            
            keywords = self._extract_keywords(text)
            
            # Prepare prompt for Ollama (same as your SUMMARIZATION_PROMPT)
            clipped = text[:3000]
            prompt = f"""You are an expert meeting summarizer. Create concise bullet points from this transcript.

TRANSCRIPT:
{clipped}

INSTRUCTIONS:
- Extract 3-5 key discussion points
- Each bullet should be a complete thought (10-20 words)
- Focus on decisions, action items, and important information
- Use third person ("The speaker discussed..." not "I discussed...")
- Be factual, avoid interpretation

OUTPUT FORMAT (JSON):
{{"bullets": ["point 1", "point 2", "point 3"]}}
"""
            # Try Ollama first
            response = self._call_ollama(prompt, max_tokens=400)
            bullets = []
            
            if response:
                # Attempt to parse JSON response
                try:
                    # Find JSON part (model may add extra text)
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = response[json_start:json_end]
                        data = json.loads(json_str)
                        bullets = data.get('bullets', [])
                    else:
                        # Fallback: split lines that look like bullet points
                        lines = response.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line.startswith('-') or line.startswith('•') or line.startswith('*'):
                                bullets.append(line.lstrip('-•* ').strip())
                except json.JSONDecodeError:
                    logger.warning("Ollama response not valid JSON, trying line parsing")
                    # Heuristic: take lines that are not empty and not too short
                    for line in response.split('\n'):
                        line = line.strip()
                        if line and len(line) > 20 and not line.startswith('{') and not line.startswith('}'):
                            bullets.append(line)
            
            # If we got bullets from Ollama, use them
            if bullets:
                logger.info(f"Ollama summarization: {len(bullets)} bullets")
                return {**state, 'bullet_points': bullets[:5], 'keywords': keywords, 'status': 'summarization_complete'}
            
            # Fallback to extractive summarization
            logger.info("Ollama not available or returned empty, using extractive fallback")
            bullets = self._extractive_fallback(text)
            return {**state, 'bullet_points': bullets, 'keywords': keywords, 'status': 'summarization_complete'}
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {**state, 'bullet_points': [], 'keywords': [], 'error': str(e), 'status': 'summarization_failed'}


# ==================== TTS AGENT ====================
class TTSAgent:
    """Agent for text-to-speech synthesis."""
    
    def __call__(self, state: AgentState) -> AgentState:
        """Convert summary to audio."""
        try:
            import pyttsx3
            
            bullets = state.get('bullet_points', [])
            if not bullets:
                return {**state, 'tts_audio_path': '', 'status': 'complete'}
            
            text = ' '.join(bullets)[:3000]
            session_id = state.get('session_id', uuid.uuid4().hex[:8])
            output_path = OUTPUT_DIR / f'summary_audio_{session_id}.mp3'
            
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
            
            logger.info(f"TTS audio saved: {output_path}")
            return {**state, 'tts_audio_path': str(output_path), 'status': 'complete'}
            
        except Exception as e:
            logger.warning(f"TTS failed: {e}")
            return {**state, 'tts_audio_path': '', 'status': 'complete'}


# ==================== LANGGRAPH WORKFLOW ====================
def create_workflow() -> StateGraph:
    """Create the LangGraph workflow with all agents."""
    
    # Initialize agents
    audio_agent = AudioProcessingAgent()
    diarization_agent = DiarizationAgent()
    transcription_agent = TranscriptionAgent()
    summarization_agent = SummarizationAgent()
    tts_agent = TTSAgent()
    speaker_attribution_agent = SpeakerAttributionAgent()
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("audio_processing", audio_agent)
    workflow.add_node("diarization", diarization_agent)
    workflow.add_node("transcription", transcription_agent)
    workflow.add_node("speaker_attribution", speaker_attribution_agent)
    workflow.add_node("summarization", summarization_agent)
    workflow.add_node("tts", tts_agent)
    
    # Define routing
    def should_continue(state: AgentState) -> str:
        if state.get('error') or state.get('status') == 'failed':
            return END
        return "continue"
    
    # Add edges
    workflow.set_entry_point("audio_processing")
    workflow.add_conditional_edges("audio_processing", should_continue, {"continue": "diarization", END: END})
    workflow.add_edge("diarization", "transcription")
    workflow.add_conditional_edges("transcription", should_continue, {"continue": "speaker_attribution", END: END})
    workflow.add_edge("speaker_attribution", "summarization")
    workflow.add_edge("summarization", "tts")
    workflow.add_edge("tts", END)
    
    return workflow.compile()


# ==================== MAIN PIPELINE ====================
class SmartVoicePipeline:
    """Main pipeline using LangGraph agents."""
    
    def __init__(self):
        self.workflow = create_workflow()
    
    def process(
        self,
        audio_path: str,
        model_size: str = 'small',
        language: str = 'auto',
        enable_diarization: bool = True,
        enable_noise_reduction: bool = True
    ) -> Dict[str, Any]:
        """Process audio through the agent workflow."""
        session_id = uuid.uuid4().hex[:8]
        
        initial_state: AgentState = {
            'audio_path': str(audio_path),
            'model_size': model_size,
            'language': language,
            'enable_diarization': enable_diarization,
            'enable_noise_reduction': enable_noise_reduction,
            'processed_audio_path': '',
            'speaker_timeline': {},
            'speaker_info': {},
            'segments': [],
            'full_transcript': '',
            'detected_language': '',
            'bullet_points': [],
            'keywords': [],
            'speaker_summaries': {},
            'tts_audio_path': '',
            'session_id': session_id,
            'error': '',
            'status': 'starting'
        }
        
        logger.info(f"Starting pipeline: session={session_id}")
        result = self.workflow.invoke(initial_state)
        
        # Save outputs
        if result.get('full_transcript'):
            self._save_outputs(result)
        
        return result
    
    def _save_outputs(self, state: Dict) -> None:
        """Save transcript and summary files."""
        session_id = state['session_id']
        
        # Save transcript
        transcript_path = OUTPUT_DIR / f'transcript_{session_id}.txt'
        formatted = self._format_transcript(state['segments'])
        transcript_path.write_text(formatted, encoding='utf-8')
        
        # Save summary JSON
        summary_path = OUTPUT_DIR / f'summary_{session_id}.json'
        summary_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'language': state['detected_language'],
            'speakers': len(state.get('speaker_info', {})) or 1,
            'segments': len(state['segments']),
            'bullets': state['bullet_points'],
            'keywords': state['keywords']
        }
        summary_path.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding='utf-8')

        # Offline-first session persistence
        try:
            try:
                from HamSabaqAi.SmartVoice.storage import SessionStore
            except ModuleNotFoundError:
                from SmartVoice.storage import SessionStore
            store = SessionStore()
            store.save_session(state, state.get('audio_path'), formatted)
        except Exception as e:
            logger.warning(f"Session persistence failed: {e}")
    
    def _format_transcript(self, segments: List[Dict]) -> str:
        """Format transcript with speaker labels."""
        lines, current_speaker = [], None
        for seg in segments:
            if seg['speaker'] != current_speaker:
                current_speaker = seg['speaker']
                lines.append(f"\n[{current_speaker}]")
            lines.append(f"  [{seg['start']:.1f}s] {seg['text']}")
        return '\n'.join(lines)
