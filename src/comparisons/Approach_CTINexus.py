# -*- coding: utf-8 -*-

import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
from sklearn.feature_extraction import text as sk_text
from sklearn.feature_extraction.text import TfidfVectorizer

DROPBOX_PATH = os.path.join(os.path.expanduser("~"), "Dropbox")
if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

TOOLS_FILE = os.path.join(DROPBOX_PATH, "tools.py")
_loaded_tools = sys.modules.get("tools")
if _loaded_tools is None or os.path.abspath(getattr(_loaded_tools, "__file__", "")) != TOOLS_FILE:
    spec = importlib.util.spec_from_file_location("tools", TOOLS_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"❌ 无法加载 Dropbox tools.py: {TOOLS_FILE}")
    _loaded_tools = importlib.util.module_from_spec(spec)
    sys.modules["tools"] = _loaded_tools
    spec.loader.exec_module(_loaded_tools)
tools = _loaded_tools

from shared_eval_backend import lookup_sample_ref, run_logged_asks

try:
    from article_io_cache_parser import KGCacheManager
except ImportError:
    print("⚠️ Warning: Could not import KGCacheManager. Caching disabled.")
    KGCacheManager = None

REPO_ROOT = Path(CURRENT_DIR).resolve().parents[1]
CTINEXUS_REPO = REPO_ROOT / "baseline" / "resources"
DEMO_DIR = str(CTINEXUS_REPO / "ctinexus_demo")
EVAL_ROOT = Path(CURRENT_DIR).resolve().parent
CTINEXUS_SHARED_CACHE_DIR = EVAL_ROOT / "GeneratedKGContent" / "Approach_CTINexus" / "_shared"
MENTION_EMBEDDING_CACHE_DIR = CTINEXUS_SHARED_CACHE_DIR / "mention_embeddings"

IE_PROMPT_TEMPLATE = """As a security analyst, your task is to analyze cyber threats by reading reports and extracting useful information in the form of subject-relation-object triplets.
The extracted triplets should adhere to the JSON format: {{"subject": "...(entity class)", "relation": "...", "object": "...(entity class)"}}.
Ensure that "subject" and "object" are limited to the following entity classes: Malware Type, Malware, Application, Campaign, System, System Feature, Organization, Time, Threat Actor, Location, Indicator Type, Indicator, Attack Pattern, Vulnerability Type, Vulnerability, Report.
Do not extract any other types of information outside these specified classes.
The subject should not be equal to the class (e.g. "Threat Actor (Threat Actor)" is invalid) or so generic it's irrelevant (e.g. "adversary (Threat Actor)"). Avoid repeating the class in the subject (e.g. "remote code execution vulnerability (Vulnerability Type)" would be better as "remote code execution (Vulnerability Type)").
Your response must be a JSON object follows the format: {{"triplets": [{{"subject": "...(entity class)", "relation": "...", "object": "...(entity class)"}}, ...]}}.
--------------------{demos_section}
Target report:

"CTI": {content}

Your response (JSON only):"""

DEMO_TEMPLATE = """
Example {index}:

'CTI': {text}

'triplets': {triplets}
--------------------"""

ET_PROMPT = """Classify the given triple's subject and object into one of the following categories:
[
    "Account", "Credential", "Tool", "Attacker", "Event", "Exploit Target",
    "Indicator": {{"File", "IP", "URL", "Domain", "Registry Key", "Hash", "Mutex", "User Agent", "Email", "Yara Rule", "SSL Certificate"}},
    "Information", "Location", "Malware",
    "Malware Characteristic": {{"Behavior", "Capability", "Feature", "Payload", "Variants"}},
    "Organization", "Infrastructure", "Time", "Vulnerability",
    "This entity cannot be classified into any of the existing types"
]
Your response must be JSON and nothing else.
---------------------
Target Triples:

{triples_section}

Your response should follow the format: {{"tagged_triples": [`tagged_triple_1`, `tagged_triple_2`, ..., `tagged_triple_n`]}}"""

LINK_PROMPT = """Target task:

"Context": {context}

"Question": What do you think is the relationship between entity "{entity1}" and entity "{entity2}"?

"predicted_triple": (JSON format with subject, relation, object)"""


LOCAL_PROMPT_SAFE_INPUT_TOKENS = 30000
LOCAL_PROMPT_MAX_CONTENT_CHARS = 200000
LOCAL_PROMPT_MIN_CONTENT_CHARS = 2000
LOCAL_PROMPT_DEFAULT_DEMO_CHARS = 220
LOCAL_PROMPT_MIN_DEMO_CHARS = 60
LOCAL_LINK_CONTEXT_CHARS = 200000

IOC_PATTERNS = {
    "date": re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b"),
    "ip": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
    "domain": re.compile(r"\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,20}\b"),
    "url": re.compile(r"\b(?:https?://|www\.)[a-zA-Z0-9-]+\.[a-zA-Z]{2,20}\S*\b"),
    "email": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,20}\b"),
    "hash_md5": re.compile(r"\b[a-fA-F0-9]{32}\b"),
    "hash_sha1": re.compile(r"\b[a-fA-F0-9]{40}\b"),
    "hash_sha256": re.compile(r"\b[a-fA-F0-9]{64}\b"),
    "hash_sha512": re.compile(r"\b[a-fA-F0-9]{128}\b"),
    "cve": re.compile(r"\bCVE-\d{4}-\d{4,7}\b"),
    "cvss": re.compile(r"\bCVSS\d\.\d\b"),
    "yara": re.compile(r"\bYARA\d{4}\b"),
}

STOP_WORDS = set(sk_text.ENGLISH_STOP_WORDS)


def _safe_json_loads(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    parsed = _safe_json_loads(text or "")
    if isinstance(parsed, dict):
        return parsed

    match = re.search(r"\{[\s\S]*\}", text or "")
    if not match:
        return None
    parsed = _safe_json_loads(match.group())
    return parsed if isinstance(parsed, dict) else None


def _clean_tfidf_text(text: str) -> str:
    pieces: List[str] = []
    for word in str(text or "").split():
        cleaned = re.sub(r"[^a-zA-Z]", " ", word).lower().strip()
        if cleaned and cleaned not in STOP_WORDS:
            pieces.append(cleaned)
    return " ".join(pieces)


def _normalize_class(value: Any) -> str:
    if isinstance(value, dict):
        keys = list(value.keys())
        if keys:
            return str(keys[0]).strip() or "Unknown"
        return "Unknown"
    text = str(value or "").strip()
    return text or "Unknown"


def _normalize_entity(entity: Any) -> Dict[str, Any]:
    if isinstance(entity, dict):
        text = str(
            entity.get("text")
            or entity.get("mention_text")
            or entity.get("entity_text")
            or entity.get("name")
            or ""
        ).strip()
        cls = _normalize_class(entity.get("class") or entity.get("mention_class") or entity.get("type"))
        return {"text": text, "class": cls}

    text = str(entity or "").strip()
    cls = "Unknown"
    match = re.match(r"^(.*?)(?:\s*\(([^()]+)\))?$", text)
    if match:
        raw_text = str(match.group(1) or "").strip()
        raw_cls = str(match.group(2) or "").strip()
        if raw_text:
            text = raw_text
        if raw_cls:
            cls = raw_cls
    return {"text": text, "class": cls}


def _entity_label(entity: Dict[str, Any]) -> Tuple[str, str]:
    return (str(entity.get("text", "") or "").strip(), _normalize_class(entity.get("class")))


def _is_full_ioc_mention(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate:
        return False
    for pattern in IOC_PATTERNS.values():
        try:
            if pattern.fullmatch(candidate):
                return True
        except Exception:
            continue
    return False


class DemoRetriever:

    def __init__(self, demo_dir: str = DEMO_DIR, retriever_type: str = "tfidf", k: int = 3):
        self.demo_dir = demo_dir
        self.retriever_type = str(retriever_type or "tfidf").lower()
        self.k = max(1, int(k or 1))
        self.demos = self._load_demos()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        self.cleaned_docs: List[str] = []
        if self.retriever_type in {"tfidf", "knn"} and self.demos:
            self._build_index()

    def _load_demos(self) -> List[Dict[str, Any]]:
        demos: List[Dict[str, Any]] = []
        if not os.path.isdir(self.demo_dir):
            print(f"⚠️ [CTINexus] demo 目录不存在: {self.demo_dir}")
            return demos
        for filename in sorted(os.listdir(self.demo_dir)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(self.demo_dir, filename)
            try:
                data = json.load(open(path, "r", encoding="utf-8"))
            except Exception as exc:
                print(f"⚠️ [CTINexus] demo 读取失败 {filename}: {exc}")
                continue
            demos.append(
                {
                    "filename": filename,
                    "text": str(data.get("text", "") or ""),
                    "triplets": data.get("explicit_triplets", []) or [],
                }
            )
        return demos

    def _build_index(self) -> None:
        self.cleaned_docs = [_clean_tfidf_text(item["text"]) for item in self.demos]
        self.vectorizer = TfidfVectorizer()
        self.doc_matrix = self.vectorizer.fit_transform(self.cleaned_docs)
        print(f"📚 [CTINexus] TF-IDF demo index 已构建: demos={len(self.demos)}")

    def retrieve(self, query_text: str) -> Tuple[List[Tuple[str, List[Dict[str, Any]]]], List[Tuple[str, float]]]:
        if not self.demos:
            return [], []

        if self.retriever_type == "fixed":
            selected = self.demos[: self.k]
            return (
                [(item["text"], item["triplets"]) for item in selected],
                [(item["filename"], 1.0) for item in selected],
            )
        if self.retriever_type == "random":
            selected = random.sample(self.demos, min(self.k, len(self.demos)))
            return (
                [(item["text"], item["triplets"]) for item in selected],
                [(item["filename"], 0.0) for item in selected],
            )

        if self.vectorizer is None or self.doc_matrix is None:
            self._build_index()
        query_clean = _clean_tfidf_text(query_text)
        query_vec = self.vectorizer.transform([query_clean])
        sims = (self.doc_matrix @ query_vec.T).toarray().reshape(-1)
        top_indices = np.argsort(sims)[::-1][: self.k]
        demos = [(self.demos[idx]["text"], self.demos[idx]["triplets"]) for idx in top_indices]
        info = [(self.demos[idx]["filename"], float(sims[idx])) for idx in top_indices]
        return demos, info


class Merger:

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        embedding_model: str = "text-embedding-3-small",
        cache_dir: Path = MENTION_EMBEDDING_CACHE_DIR,
    ):
        self.threshold = float(similarity_threshold)
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_lock = threading.Lock()

    def _embedding_cache_path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.embedding_model}\n{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _load_embedding(self, text: str) -> Optional[List[float]]:
        path = self._embedding_cache_path(text)
        if not path.exists():
            return None
        try:
            payload = json.load(open(path, "r", encoding="utf-8"))
        except Exception:
            return None
        if payload.get("model") != self.embedding_model or payload.get("text") != text:
            return None
        emb = payload.get("embedding")
        return emb if isinstance(emb, list) else None

    def _save_embedding(self, text: str, embedding: List[float]) -> None:
        path = self._embedding_cache_path(text)
        payload = {
            "model": self.embedding_model,
            "text": text,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "embedding": embedding,
        }
        with self._cache_lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)

    def _fetch_missing_embeddings(self, missing_texts: List[str]) -> Dict[str, Optional[List[float]]]:
        results: Dict[str, Optional[List[float]]] = {}
        if not missing_texts:
            return results

        worker_count = min(8, len(missing_texts))
        print(f"  🔁 [CTINexus] EM 拉取缺失 embeddings: {len(missing_texts)} 条, workers={worker_count}")

        def _fetch(text: str) -> Tuple[str, Optional[List[float]]]:
            try:
                embedding = tools.get_embedding(text, model=self.embedding_model)
                if embedding:
                    self._save_embedding(text, embedding)
                return text, embedding
            except Exception as exc:
                print(f"  ⚠️ [CTINexus] mention embedding 失败: {text[:80]} -> {exc}")
                return text, None

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ctinexus_embed") as executor:
            future_map = {executor.submit(_fetch, text): text for text in missing_texts}
            for future in as_completed(future_map):
                text, embedding = future.result()
                results[text] = embedding
        return results

    def _get_embeddings(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        embeddings: Dict[str, Optional[List[float]]] = {}
        missing: List[str] = []
        for text in texts:
            cached = self._load_embedding(text)
            if cached is not None:
                embeddings[text] = cached
            else:
                missing.append(text)
        embeddings.update(self._fetch_missing_embeddings(missing))
        return embeddings

    def _cluster_mentions(self, mention_texts: List[str], embeddings: Dict[str, Optional[List[float]]]) -> List[Set[str]]:
        active = [text for text in mention_texts if embeddings.get(text) is not None]
        inactive = [text for text in mention_texts if embeddings.get(text) is None]
        if not active:
            return [{text} for text in mention_texts]

        matrix = np.array([embeddings[text] for text in active], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = matrix / norms
        sim_matrix = normalized @ normalized.T

        graph = nx.Graph()
        graph.add_nodes_from(active)
        for idx, text in enumerate(active):
            for jdx in range(idx + 1, len(active)):
                if float(sim_matrix[idx, jdx]) >= self.threshold:
                    graph.add_edge(text, active[jdx])

        components: List[Set[str]] = [set(comp) for comp in nx.connected_components(graph)]
        components.extend({text} for text in inactive)
        return components

    def merge_ea(self, aligned_triplets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        mention_buckets: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for triplet in aligned_triplets:
            for role in ("subject", "object"):
                node = triplet.get(role) or {}
                mention_buckets[str(node.get("mention_class", "Unknown"))][str(node.get("mention_text", ""))].append(node)

        next_entity_id = 0
        for mention_class, mentions in mention_buckets.items():
            mention_texts = [text for text in mentions.keys() if text]
            embeddings = self._get_embeddings(mention_texts)
            for component in self._cluster_mentions(mention_texts, embeddings):
                rep = max(component, key=lambda item: (len(item), item))
                entity_id = next_entity_id
                next_entity_id += 1
                component_texts = sorted(component)
                for mention_text in component_texts:
                    merged_others = [text for text in component_texts if text != mention_text]
                    for node in mentions.get(mention_text, []):
                        node["entity_id"] = entity_id
                        node["entity_text"] = rep
                        node["mention_merged"] = merged_others
                        node["mention_class"] = mention_class
        return aligned_triplets, next_entity_id


class Linker:

    def __init__(self, llm_func):
        self.llm_func = llm_func

    def _parse_link_response(self, text: str) -> Optional[Dict[str, Any]]:
        payload = _extract_json_payload(text or "")
        if not isinstance(payload, dict):
            return None
        triple = payload.get("predicted_triple", payload)
        if not isinstance(triple, dict):
            return None
        subject = str(triple.get("subject", "") or "").strip()
        relation = str(triple.get("relation", "") or "").strip()
        object_ = str(triple.get("object", "") or "").strip()
        if not relation:
            return None
        return {"subject": subject, "relation": relation, "object": object_}

    def link(self, triplets: List[Dict[str, Any]], context: str) -> List[Dict[str, Any]]:
        entity_graph = nx.Graph()
        entity_info: Dict[int, Dict[str, Any]] = {}

        for triplet in triplets:
            if not isinstance(triplet, dict):
                continue
            subject = triplet.get("subject") or {}
            object_ = triplet.get("object") or {}
            if not isinstance(subject, dict) or not isinstance(object_, dict):
                continue
            sid = subject.get("entity_id")
            oid = object_.get("entity_id")
            if sid is None or oid is None:
                continue
            entity_graph.add_edge(int(sid), int(oid))
            entity_info[int(sid)] = dict(subject)
            entity_info[int(oid)] = dict(object_)

        subgraphs = list(nx.connected_components(entity_graph))
        if len(subgraphs) < 2:
            return triplets

        def _main_node(subgraph: Set[int]) -> int:
            sub_g = entity_graph.subgraph(subgraph)
            return max(dict(sub_g.degree()).items(), key=lambda item: (item[1], item[0]))[0]

        topic_subgraph = max(subgraphs, key=lambda item: len(item))
        topic_node_id = _main_node(topic_subgraph)
        topic_node = entity_info[topic_node_id]
        new_triplets = list(triplets)

        for subgraph in subgraphs:
            if topic_node_id in subgraph:
                continue
            main_node_id = _main_node(subgraph)
            main_node = entity_info[main_node_id]
            prompt = LINK_PROMPT.format(
                context=str(context or "")[:LOCAL_LINK_CONTEXT_CHARS],
                entity1=main_node.get("text", ""),
                entity2=topic_node.get("text", ""),
            )
            try:
                response = self.llm_func(prompt)
                parsed = self._parse_link_response(response)
                if not parsed:
                    continue
                pred_sub = parsed["subject"]
                pred_obj = parsed["object"]
                pred_rel = parsed["relation"]
                if pred_sub == main_node.get("text") and pred_obj == topic_node.get("text"):
                    new_triplets.append(
                        {
                            "subject": dict(main_node),
                            "relation": pred_rel,
                            "object": dict(topic_node),
                        }
                    )
                elif pred_sub == topic_node.get("text") and pred_obj == main_node.get("text"):
                    new_triplets.append(
                        {
                            "subject": dict(topic_node),
                            "relation": pred_rel,
                            "object": dict(main_node),
                        }
                    )
                else:
                    print(
                        "  ⚠️ [CTINexus] Link hallucination 已丢弃: "
                        f"{pred_sub} - {pred_rel} - {pred_obj}"
                    )
            except Exception as exc:
                print(f"  ⚠️ [CTINexus] Link 预测失败: {exc}")
        return new_triplets


class CTINexusMethod:
    def __init__(
        self,
        model: str = "gpt-5-nano",
        token: int = 16 * 1024,
        temp: float = 0.3,
        retriever_type: str = "tfidf",
        k: int = 3,
        use_link_stage: bool = True,
        use_cloud_or_vllm: str = "cloud",
        vllm_model_path: Optional[str] = None,
        shared_llm_backend: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        max_batch_workers: int = 8,
        ie_demo_text_chars: int = LOCAL_PROMPT_DEFAULT_DEMO_CHARS,
        enable_method_cache: bool = False,
        **kwargs,
    ):
        self.shared_llm_backend = dict(shared_llm_backend or {})
        self.runtime_context = runtime_context or {}
        self.use_cloud_or_vllm = "vllm" if self.shared_llm_backend.get("enabled") else str(use_cloud_or_vllm).lower()
        self.vllm_model_path = self.shared_llm_backend.get("model_path", vllm_model_path)
        self.model = self.shared_llm_backend.get("model", (model if self.use_cloud_or_vllm == "cloud" else "local"))
        self.token = token
        self.temp = temp
        self.max_batch_workers = max(1, int(max_batch_workers or 1))
        self.ie_demo_text_chars = max(40, int(ie_demo_text_chars or LOCAL_PROMPT_DEFAULT_DEMO_CHARS))
        self.use_link_stage = bool(use_link_stage)
        self.retriever = DemoRetriever(retriever_type=retriever_type, k=k)
        self.merger = Merger()
        self.name = "CTINexus"
        self.ask_check_history_cache = self.shared_llm_backend.get("check_history_cache", True)
        self.ask_vllm_smart_mode = self.shared_llm_backend.get("smart_mode", True)
        self.ask_max_workers_vllm = self.shared_llm_backend.get("max_workers_vllm", "auto")
        self.ask_prompt_send_weight_vllm = self.shared_llm_backend.get(
            "prompt_send_weight_vllm",
            {"super": 2, "ultra": 4, "normal": 1},
        )
        self.ask_vllm_server_name = self.shared_llm_backend.get("vllm_server_name")
        self._grid_current_stage = "generate"
        self._grid_current_sample_ref = {}

        self.vllm_manager = None
        if self.use_cloud_or_vllm == "vllm" and not self.shared_llm_backend.get("enabled"):
            if not vllm_model_path:
                raise ValueError("❌ 使用 vLLM 模式时必须指定 vllm_model_path 参数")
            from vllm_environment_setup import VLLMEnvironmentManager

            self.vllm_manager = VLLMEnvironmentManager(vllm_model_path)
            print(f"🖥️ CTINexus vLLM 模式: {vllm_model_path}")
        elif self.use_cloud_or_vllm == "vllm":
            print(f"🖥️ CTINexus 共享三机 vLLM 模式: {self.vllm_model_path}")
        else:
            print(f"☁️ CTINexus 云端模式: {model}")

        
        
        
        if KGCacheManager and enable_method_cache:
            self.cache = KGCacheManager(os.path.basename(__file__), method_file_path=os.path.abspath(__file__))
        else:
            self.cache = None

    def _estimate_prompt_tokens(self, text: str) -> int:
        try:
            return max(1, int(tools.tokenlen(text or "")))
        except Exception:
            return max(1, len(text or "") // 4)

    def _build_ie_prompt(
        self,
        content: str,
        demos: List[Tuple[str, List[Dict[str, Any]]]],
        *,
        content_limit_override: Optional[int] = None,
        demo_text_limit_override: Optional[int] = None,
        max_demos_override: Optional[int] = None,
    ) -> Tuple[str, Dict[str, int]]:
        demos_to_use = list(demos[: max(0, int(max_demos_override or len(demos)))])
        demo_text_limit = self.ie_demo_text_chars if demo_text_limit_override is None else int(demo_text_limit_override)
        content_limit = len(content) if content_limit_override is None else min(len(content), int(content_limit_override))
        content_limit = min(content_limit, LOCAL_PROMPT_MAX_CONTENT_CHARS)

        def _render() -> str:
            demo_section = ""
            for idx, (demo_text, demo_triplets) in enumerate(demos_to_use):
                demo_section += DEMO_TEMPLATE.format(
                    index=idx + 1,
                    text=str(demo_text or "")[:demo_text_limit],
                    triplets=json.dumps({"triplets": demo_triplets}, ensure_ascii=False, separators=(",", ":")),
                )
            return IE_PROMPT_TEMPLATE.format(demos_section=demo_section, content=content[:content_limit])

        prompt = _render()
        estimated_tokens = self._estimate_prompt_tokens(prompt)
        original_tokens = estimated_tokens

        while estimated_tokens > LOCAL_PROMPT_SAFE_INPUT_TOKENS:
            changed = False
            if demo_text_limit > LOCAL_PROMPT_MIN_DEMO_CHARS:
                new_demo_limit = max(LOCAL_PROMPT_MIN_DEMO_CHARS, int(demo_text_limit * 0.7))
                changed = new_demo_limit < demo_text_limit
                demo_text_limit = new_demo_limit
            elif len(demos_to_use) > 0:
                demos_to_use = demos_to_use[:-1]
                changed = True
            if not changed:
                break
            prompt = _render()
            estimated_tokens = self._estimate_prompt_tokens(prompt)

        if estimated_tokens < original_tokens:
            print(
                f"  ✂️ [CTINexus] IE prompt 收缩: {original_tokens} -> {estimated_tokens} tokens "
                f"(content_chars={content_limit}, demo_chars={demo_text_limit}, demos={len(demos_to_use)})"
            )
        elif estimated_tokens > LOCAL_PROMPT_SAFE_INPUT_TOKENS:
            print(
                f"  ⚠️ [CTINexus] IE prompt 仍较长 ({estimated_tokens} tokens)，"
                f"但 v3 按全文公平输入原则保留完整文章，不再裁正文"
            )

        return prompt, {
            "content_limit": content_limit,
            "demo_text_limit": demo_text_limit,
            "demo_count": len(demos_to_use),
            "estimated_tokens": estimated_tokens,
        }

    def _build_et_prompt(self, triplets: List[Dict[str, Any]]) -> Optional[str]:
        triples_section = ""
        for triplet in triplets:
            triples_section += (
                f'"triple": {json.dumps(triplet, ensure_ascii=False, separators=(",", ":"))}\n'
                '"tagged_triple": """..."""\n---\n'
            )
        prompt = ET_PROMPT.format(triples_section=triples_section)
        estimated_tokens = self._estimate_prompt_tokens(prompt)
        if self.use_cloud_or_vllm == "vllm" and estimated_tokens > LOCAL_PROMPT_SAFE_INPUT_TOKENS:
            print(
                f"  ⚠️ [CTINexus] ET prompt 约 {estimated_tokens} tokens，超过本地预算 "
                f"{LOCAL_PROMPT_SAFE_INPUT_TOKENS}，回退到 IE 结果"
            )
            return None
        return prompt

    def _call_llm(self, prompt: str) -> str:
        if self.vllm_manager:
            self.vllm_manager.ensure_ready()
        messages = [{"role": "user", "content": prompt}]
        prompt_metadata = [dict(self._grid_current_sample_ref, stage=self._grid_current_stage)]
        results = run_logged_asks(
            [messages],
            model=self.model,
            token=self.token,
            temp=self.temp,
            runtime_context=self.runtime_context,
            phase="generate",
            prompt_metadata_list=prompt_metadata,
            check_history_cache=self.ask_check_history_cache,
            VllmSmartMode=self.ask_vllm_smart_mode,
            max_workers_Vllm=self.ask_max_workers_vllm,
            prompt_send_weight_VllmNotSmartMode=self.ask_prompt_send_weight_vllm,
            vllm_server_name=self.ask_vllm_server_name,
        )
        return results[0] if results else ""

    def _normalize_triplets(self, raw_triplets: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(raw_triplets, list):
            return normalized
        for triplet in raw_triplets:
            if isinstance(triplet, str):
                parsed = _safe_json_loads(triplet)
                triplet = parsed if isinstance(parsed, dict) else None
            if not isinstance(triplet, dict):
                continue
            subject = _normalize_entity(triplet.get("subject"))
            object_ = _normalize_entity(triplet.get("object"))
            relation = str(triplet.get("relation", "") or "").strip()
            if not subject["text"] or not object_["text"] or not relation:
                continue
            normalized.append({"subject": subject, "relation": relation, "object": object_})
        return normalized

    def _preprocess_ea(self, typed_triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
        aligned_triplets: List[Dict[str, Any]] = []
        mention_id_map: Dict[str, int] = {}
        next_mention_id = 0

        for triplet in typed_triplets:
            new_triplet = {"relation": triplet["relation"]}
            for role in ("subject", "object"):
                entity = dict(triplet[role])
                mention_text = str(entity.get("text", "") or "").strip()
                if mention_text not in mention_id_map:
                    mention_id_map[mention_text] = next_mention_id
                    next_mention_id += 1
                new_triplet[role] = {
                    "mention_id": mention_id_map[mention_text],
                    "mention_text": mention_text,
                    "mention_class": _normalize_class(entity.get("class")),
                    "entity_id": mention_id_map[mention_text],
                    "entity_text": mention_text,
                    "mention_merged": [],
                }
            aligned_triplets.append(new_triplet)

        return {"aligned_triplets": aligned_triplets, "mentions_num": next_mention_id}

    def _postprocess_ioc(self, aligned_triplets: List[Dict[str, Any]], entity_num: int) -> Tuple[List[Dict[str, Any]], int]:
        entity_to_nodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for triplet in aligned_triplets:
            for role in ("subject", "object"):
                node = triplet.get(role) or {}
                entity_to_nodes[int(node.get("entity_id", -1))].append(node)

        next_entity_id = int(entity_num)
        for entity_id, nodes in entity_to_nodes.items():
            mention_texts = sorted({str(node.get("mention_text", "") or "").strip() for node in nodes if node.get("mention_text")})
            if len(mention_texts) <= 1:
                continue
            full_iocs = [text for text in mention_texts if _is_full_ioc_mention(text)]
            if len(full_iocs) != len(mention_texts):
                continue

            mention_to_entity: Dict[str, int] = {}
            for mention_text in mention_texts:
                mention_to_entity[mention_text] = next_entity_id
                next_entity_id += 1

            for node in nodes:
                mention_text = str(node.get("mention_text", "") or "").strip()
                node["entity_id"] = mention_to_entity[mention_text]
                node["entity_text"] = mention_text
                node["mention_merged"] = []
        return aligned_triplets, next_entity_id

    def _aligned_to_triplets(self, aligned_triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        triplets: List[Dict[str, Any]] = []
        for aligned in aligned_triplets:
            subject = aligned.get("subject") or {}
            object_ = aligned.get("object") or {}
            relation = str(aligned.get("relation", "") or "").strip()
            if not relation:
                continue
            triplets.append(
                {
                    "subject": {
                        "text": str(subject.get("entity_text", subject.get("mention_text", "")) or "").strip(),
                        "class": _normalize_class(subject.get("mention_class")),
                        "entity_id": int(subject.get("entity_id", 0)),
                    },
                    "relation": relation,
                    "object": {
                        "text": str(object_.get("entity_text", object_.get("mention_text", "")) or "").strip(),
                        "class": _normalize_class(object_.get("mention_class")),
                        "entity_id": int(object_.get("entity_id", 0)),
                    },
                }
            )
        return self._dedupe_triplets(triplets)

    def _dedupe_triplets(self, triplets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str, str, str]] = set()
        for triplet in triplets:
            subject = triplet.get("subject") or {}
            object_ = triplet.get("object") or {}
            key = (
                str(subject.get("text", "") or "").strip(),
                _normalize_class(subject.get("class")),
                str(triplet.get("relation", "") or "").strip(),
                str(object_.get("text", "") or "").strip(),
                _normalize_class(object_.get("class")),
            )
            if not key[0] or not key[2] or not key[3]:
                continue
            if key in seen:
                continue
            seen.add(key)
            deduped.append(triplet)
        return deduped

    def _format_output(self, triplets: List[Dict[str, Any]]) -> str:
        entities: List[Dict[str, Any]] = []
        relations: List[Dict[str, Any]] = []
        seen_entities: Set[Tuple[str, str]] = set()
        for triplet in self._dedupe_triplets(triplets):
            subject = triplet["subject"]
            object_ = triplet["object"]
            relation = str(triplet.get("relation", "") or "").strip()
            for entity in (subject, object_):
                label = _entity_label(entity)
                if not label[0]:
                    continue
                if label not in seen_entities:
                    entities.append({"name": label[0], "type": label[1]})
                    seen_entities.add(label)
            if relation:
                relations.append(
                    {
                        "sub": str(subject.get("text", "") or "").strip(),
                        "rel": relation,
                        "obj": str(object_.get("text", "") or "").strip(),
                    }
                )
        return json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False)

    def generate(self, content: str) -> str:
        print(f"🚀 [{self.name}] 开始处理...")
        self._grid_current_sample_ref = lookup_sample_ref(self.runtime_context, content)

        if self.cache:
            cached_result = self.cache.load(content)
            if cached_result is not None:
                print(f"✨ [CTINexus] 命中缓存: chars={len(content)}")
                self._grid_current_sample_ref = {}
                return cached_result

        latest_valid_triplets: List[Dict[str, Any]] = []

        try:
            demos, demo_info = self.retriever.retrieve(content)
            if demo_info:
                preview = ", ".join(f"{name}:{score:.3f}" for name, score in demo_info[:3])
                print(f"  📚 [CTINexus] demo top-k: {preview}")
            prompt, _ = self._build_ie_prompt(content, demos, max_demos_override=len(demos))
            self._grid_current_stage = "ie"
            ie_resp = self._call_llm(prompt)
            ie_payload = _extract_json_payload(ie_resp or "")
            latest_valid_triplets = self._normalize_triplets((ie_payload or {}).get("triplets", []))
            if latest_valid_triplets:
                print(f"  ✅ Stage 1 (IE): {len(latest_valid_triplets)} triplets")
            else:
                print("  ⚠️ Stage 1 (IE): 无可用 triplets，返回空图")
                result = self._format_output([])
                if self.cache:
                    self.cache.save(content, result)
                self._grid_current_sample_ref = {}
                self._grid_current_stage = "generate"
                return result
        except Exception as exc:
            print(f"  ❌ Stage 1 (IE) 失败: {exc}")
            self._grid_current_sample_ref = {}
            self._grid_current_stage = "generate"
            raise

        try:
            et_prompt = self._build_et_prompt(latest_valid_triplets)
            if et_prompt:
                self._grid_current_stage = "et"
                et_resp = self._call_llm(et_prompt)
                et_payload = _extract_json_payload(et_resp or "")
                tagged_triplets = self._normalize_triplets((et_payload or {}).get("tagged_triples", []))
                if tagged_triplets:
                    latest_valid_triplets = tagged_triplets
                    print(f"  ✅ Stage 2 (ET): {len(latest_valid_triplets)} triplets")
                else:
                    print("  ⚠️ Stage 2 (ET): 解析失败，回退 IE")
        except Exception as exc:
            print(f"  ❌ Stage 2 (ET) 失败: {exc}，回退 IE")

        try:
            ea_payload = self._preprocess_ea(latest_valid_triplets)
            aligned_triplets = ea_payload["aligned_triplets"]
            aligned_triplets, entity_num = self.merger.merge_ea(aligned_triplets)
            aligned_triplets, entity_num = self._postprocess_ioc(aligned_triplets, entity_num)
            latest_valid_triplets = self._aligned_to_triplets(aligned_triplets)
            print(f"  ✅ Stage 2.5 (EM/IOC): entities={entity_num}, triplets={len(latest_valid_triplets)}")
        except Exception as exc:
            print(f"  ❌ Stage 2.5 (EM/IOC) 失败: {exc}，回退 ET")

        if self.use_link_stage:
            try:
                self._grid_current_stage = "link"
                linker = Linker(self._call_llm)
                linked_triplets = linker.link(latest_valid_triplets, content)
                linked_triplets = self._dedupe_triplets(linked_triplets)
                if linked_triplets:
                    latest_valid_triplets = linked_triplets
                    print(f"  ✅ Stage 3 (Link): triplets={len(latest_valid_triplets)}")
            except Exception as exc:
                print(f"  ❌ Stage 3 (Link) 失败: {exc}，回退 EM/IOC")

        final_output = self._format_output(latest_valid_triplets)
        if self.cache:
            self.cache.save(content, final_output)
        self._grid_current_sample_ref = {}
        self._grid_current_stage = "generate"
        return final_output

    def batch_generate(self, contents: List[str], num_workers: int = 1, **kwargs) -> List[Dict[str, Any]]:
        if not contents:
            return []

        requested_workers = max(1, int(num_workers or 1))
        worker_count = max(1, min(len(contents), requested_workers, self.max_batch_workers))
        print(f"🚀 [{self.name}] batch_generate: samples={len(contents)}, workers={worker_count}")

        if worker_count == 1:
            return [json.loads(self.generate(content)) for content in contents]

        results: List[Optional[Dict[str, Any]]] = [None] * len(contents)

        def _run_single(idx: int, content: str) -> Tuple[int, Dict[str, Any]]:
            try:
                return idx, json.loads(self.generate(content))
            except Exception as exc:
                print(f"  ⚠️ [CTINexus] 样本 {idx + 1} 失败: {exc}")
                return idx, {"entities": [], "relations": [], "error": str(exc)}

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="ctinexus_batch") as executor:
            future_map = {executor.submit(_run_single, idx, content): idx for idx, content in enumerate(contents)}
            completed = 0
            for future in as_completed(future_map):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                if completed % 5 == 0 or completed == len(contents):
                    print(f"  🔄 [{self.name}] batch 进度: {completed}/{len(contents)}")

        return [item if item is not None else {"entities": [], "relations": [], "error": "Unknown gap"} for item in results]


Method = CTINexusMethod


if __name__ == "__main__":
    method = Method()
    print(method.generate("APT29 used DNS tunneling and deployed SUNBURST against SolarWinds Orion."))
