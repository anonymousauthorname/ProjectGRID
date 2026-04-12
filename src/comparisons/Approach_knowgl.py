# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from threading import Lock
from typing import Dict, List, Optional, Tuple

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from related_work_bridge import dedupe_relations
from non_llm_dual_gpu_inference import parallel_run_on_devices, resolve_worker_devices


KNOWGL_MODEL_ID = "ibm-research/knowgl-large"


def _parse_knowgl_output(text: str) -> List[Dict[str, str]]:
    triplets: List[Dict[str, str]] = []

    pattern = r"\[\(([^)]+)\)\|([^|]+)\|\(([^)]+)\)\]"
    matches = re.findall(pattern, text)
    for sub_group, rel, obj_group in matches:
        sub = sub_group.split("#")[0].strip()
        obj = obj_group.split("#")[0].strip()
        rel = rel.strip()
        if sub and rel and obj:
            triplets.append({"sub": sub, "rel": rel, "obj": obj})

    if triplets:
        return triplets

    fallback_pattern = r"\(([^$\)]+)\$([^$\)]+)\$([^$\)]+)\)"
    for sub, rel, obj in re.findall(fallback_pattern, text):
        sub = sub.strip()
        rel = rel.strip()
        obj = obj.strip()
        if sub and rel and obj:
            triplets.append({"sub": sub, "rel": rel, "obj": obj})
    return triplets


class KnowGLMethod:
    _models = {}
    _tokenizers = {}
    _devices = {}
    _load_lock = Lock()

    def __init__(
        self,
        model_id: str = KNOWGL_MODEL_ID,
        encoder_max_length: int = 256,
        generation_max_length: int = 256,
        
        inference_batch_size: int = 64,
        num_beams: int = 1,
        max_devices: int = 2,
        preferred_devices: Optional[List[str]] = None,
        **kwargs,
    ):
        self.name = "KnowGL"
        self.model_id = model_id
        self.encoder_max_length = encoder_max_length
        self.generation_max_length = generation_max_length
        self.inference_batch_size = inference_batch_size
        self.num_beams = max(1, int(num_beams))
        self.max_devices = max(1, int(max_devices or 1))
        self.preferred_devices = list(preferred_devices or [])
        self._force_cpu_fallback = False
        print(f"✅ {self.name} 初始化 (model={self.model_id})")

    def _resolve_device_names(self) -> List[str]:
        return resolve_worker_devices(self.preferred_devices, self.max_devices)

    def _load_model(self, device_name: Optional[str] = None):
        device_name = device_name or self._resolve_device_names()[0]
        if device_name in KnowGLMethod._models:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        with KnowGLMethod._load_lock:
            if device_name in KnowGLMethod._models:
                return
            device = torch.device(device_name if torch.cuda.is_available() else "cpu")
            print(f"  📥 加载 {self.model_id} 到 {device} ...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
            model.to(device)
            model.eval()
            KnowGLMethod._tokenizers[device_name] = tokenizer
            KnowGLMethod._models[device_name] = model
            KnowGLMethod._devices[device_name] = device
            print(f"  ✅ {self.model_id} 加载完成 ({device})")

    def _split_sentences(self, content: str) -> List[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
        if not sentences and content.strip():
            sentences = [content.strip()]
        cleaned_sentences: List[str] = []
        for sentence in sentences:
            sentence = re.sub(r"\s+", " ", sentence).strip()
            if len(sentence) < 15:
                continue
            
            
            sentence = sentence.replace("\x00", " ").replace("\ufeff", " ")
            cleaned_sentences.append(sentence[:2400])
        return cleaned_sentences

    def _reset_device_state(self, device_name: str) -> None:
        try:
            import gc
            import torch

            model = KnowGLMethod._models.pop(device_name, None)
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
            KnowGLMethod._tokenizers.pop(device_name, None)
            KnowGLMethod._devices.pop(device_name, None)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as exc:
            print(f"  ⚠️ [{self.name}|{device_name}] 设备重置失败: {exc}")

    def _safe_generate_batch(self, texts: List[str], device_name: str) -> List[Tuple[str, List[Dict[str, str]]]]:
        actual_device = "cpu" if self._force_cpu_fallback else device_name
        try:
            return self._generate_batch(texts, actual_device)
        except Exception as exc:
            err = str(exc)
            print(f"  ⚠️ [{self.name}|{actual_device}] batch={len(texts)} 推理失败，准备降级重试: {err[:240]}")
            if actual_device != "cpu":
                self._force_cpu_fallback = True
                self._reset_device_state(actual_device)
                print(f"  🔁 [{self.name}] 后续批次统一切到 CPU 兜底")
                return self._generate_batch(texts, "cpu")
            raise

    def _generate_batch(self, texts: List[str], device_name: str) -> List[Tuple[str, List[Dict[str, str]]]]:
        import torch

        self._load_model(device_name)
        tokenizer = KnowGLMethod._tokenizers[device_name]
        model = KnowGLMethod._models[device_name]
        device = KnowGLMethod._devices[device_name]

        inputs = tokenizer(
            texts,
            max_length=self.encoder_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                num_beams=self.num_beams,
                max_length=self.generation_max_length,
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [(text, _parse_knowgl_output(text)) for text in decoded]

    def _run_sentence_chunk_resilient(
        self,
        device_name: str,
        chunk: List[Tuple[int, str]],
    ) -> List[Tuple[int, str, List[Dict[str, str]]]]:
        if not chunk:
            return []
        try:
            decoded_and_triplets = self._safe_generate_batch([text for _, text in chunk], device_name=device_name)
            return [
                (article_idx, decoded, triplets)
                for (article_idx, _), (decoded, triplets) in zip(chunk, decoded_and_triplets)
            ]
        except Exception as exc:
            if len(chunk) == 1:
                article_idx, text = chunk[0]
                print(f"  ⚠️ [{self.name}|{device_name}] 单句失败，跳过: {str(exc)[:200]} | {text[:120]}")
                return [(article_idx, f"[ERROR] {exc}", [])]
            mid = max(1, len(chunk) // 2)
            left = self._run_sentence_chunk_resilient(device_name, chunk[:mid])
            right = self._run_sentence_chunk_resilient(device_name, chunk[mid:])
            return left + right

    def _run_sentence_shard(
        self,
        device_name: str,
        sentence_tasks: List[Tuple[int, str]],
        batch_size: int,
    ) -> List[Tuple[int, str, List[Dict[str, str]]]]:
        shard_results: List[Tuple[int, str, List[Dict[str, str]]]] = []
        for start in range(0, len(sentence_tasks), batch_size):
            chunk = sentence_tasks[start:start + batch_size]
            decoded_and_triplets = self._run_sentence_chunk_resilient(device_name, chunk)
            for article_idx, decoded, triplets in decoded_and_triplets:
                shard_results.append((article_idx, decoded, triplets))
            progress_device = "cpu-fallback" if self._force_cpu_fallback and device_name != "cpu" else device_name
            print(
                f"  🔄 [{self.name}|{progress_device}] sentence分片进度: "
                f"{min(start + batch_size, len(sentence_tasks))}/{len(sentence_tasks)}"
            )
        return shard_results

    def batch_generate(self, contents: List[str], num_workers: int = 1, **kwargs) -> List[Dict]:
        device_names = self._resolve_device_names()
        for device_name in device_names:
            self._load_model(device_name)
        batch_size = max(1, int(self.inference_batch_size or 1))

        sentence_tasks: List[Tuple[int, str]] = []
        article_raw_outputs = [[] for _ in contents]
        article_relations = [[] for _ in contents]

        for article_idx, content in enumerate(contents):
            for sentence in self._split_sentences(content):
                sentence_tasks.append((article_idx, sentence))

        print(
            f"🚀 [{self.name}] 批量推理开始: articles={len(contents)}, sentences={len(sentence_tasks)}, "
            f"devices={device_names}, per_device_batch={batch_size}, beams={self.num_beams}"
        )

        shard_outputs = parallel_run_on_devices(
            tasks=sentence_tasks,
            device_names=device_names,
            handler=lambda device_name, shard: self._run_sentence_shard(device_name, shard, batch_size),
            task_label="sentences",
        )
        for article_idx, decoded, triplets in shard_outputs:
            article_raw_outputs[article_idx].append(decoded)
            article_relations[article_idx].extend(triplets)

        results = []
        for raw_outputs, relations in zip(article_raw_outputs, article_relations):
            results.append(
                {
                    "entities": [],
                    "relations": dedupe_relations(relations),
                    "raw_output": "\n\n".join(raw_outputs),
                }
            )
        return results

    def generate(self, content: str) -> Dict:
        return self.batch_generate([content], num_workers=1)[0]

    def cleanup(self) -> None:
        try:
            import gc
            import torch

            for model in list(KnowGLMethod._models.values()):
                try:
                    model.to("cpu")
                except Exception:
                    pass
            KnowGLMethod._models.clear()
            KnowGLMethod._tokenizers.clear()
            KnowGLMethod._devices.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            print(f"  🧹 [{self.name}] 已释放模型与显存")
        except Exception as exc:
            print(f"  ⚠️ [{self.name}] cleanup 失败: {exc}")


Method = KnowGLMethod
