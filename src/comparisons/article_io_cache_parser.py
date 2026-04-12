# -*- coding: utf-8 -*-

import os
import sys
import json
import hashlib
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


DROPBOX_PATH = os.path.join(os.path.expanduser("~"), 'Dropbox')
if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)

if SCRIPT_DIR := os.path.dirname(os.path.abspath(__file__)):
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from shared_eval_backend import lookup_sample_ref
from src import tools_prompt_nano as local_tools_prompt


# ================================================================================

# ================================================================================




EVAL_ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CACHE_DIR = os.path.join(EVAL_ROOT_DIR, "GeneratedKGContent", "_LegacyKGCache")


def file_hash(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "file_not_found"
    
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def content_hash(content: str) -> str:
    if not isinstance(content, str):
        content = str(content)
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def combined_hash(py_hash: str, input_hash: str) -> str:
    return f"{py_hash}_{input_hash}"


class KGCacheManager:
    
    def __init__(
        self,
        method_name: str,
        method_file_path: str = None,
        cache_dir: str = None,
        auto_create: bool = True
    ):
        self.method_name = method_name
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.method_dir = os.path.join(self.cache_dir, method_name)
        
        
        if method_file_path:
            self.method_file_path = method_file_path
        else:
            
            self.method_file_path = os.path.join(SCRIPT_DIR, method_name)
        
        self.py_hash = file_hash(self.method_file_path)
        
        if auto_create:
            os.makedirs(self.method_dir, exist_ok=True)
        
        print(f"  📦 KG缓存管理器: {method_name} (py_hash={self.py_hash})")
    
    def _get_cache_path(self, content: str) -> str:
        c_hash = content_hash(content)
        full_hash = combined_hash(self.py_hash, c_hash)
        return os.path.join(self.method_dir, f"{full_hash}.json")
    
    def exists(self, content: str) -> bool:
        return os.path.exists(self._get_cache_path(content))
    
    def load(self, content: str) -> Optional[Dict[str, Any]]:
        cache_path = self._get_cache_path(content)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                
                cached_py_hash = data.get('py_hash', '')
                if cached_py_hash and cached_py_hash != self.py_hash:
                    
                    return None
                
                
                return data.get('result', data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠️ 缓存读取失败: {cache_path} ({e})")
            return None
    
    def save(self, content: str, result: Dict[str, Any]) -> str:
        cache_path = self._get_cache_path(content)
        
        c_hash = content_hash(content)
        
        
        cache_data = {
            "py_hash": self.py_hash,
            "content_hash": c_hash,
            "method": self.method_name,
            "timestamp": datetime.now().isoformat(),
            "content_preview": content[:200] + "..." if len(content) > 200 else content,
            "result": result
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"  ⚠️ 缓存保存失败: {cache_path} ({e})")
        
        return cache_path
    
    def batch_check(self, contents: List[str]) -> Tuple[Dict[int, Dict], List[int]]:
        cached_indices = {}
        uncached_indices = []
        
        for i, content in enumerate(contents):
            cache_path = self._get_cache_path(content)
            if os.path.exists(cache_path):
                cached_indices[i] = cache_path
            else:
                uncached_indices.append(i)
        
        return cached_indices, uncached_indices
    
    def batch_load(self, contents: List[str]) -> Tuple[Dict[int, Dict], List[int]]:
        cached_results = {}
        uncached_indices = []
        
        for i, content in enumerate(contents):
            result = self.load(content)
            if result is not None:
                cached_results[i] = result
            else:
                uncached_indices.append(i)
        
        return cached_results, uncached_indices
    
    def batch_save(self, contents: List[str], results: List[Dict[str, Any]]) -> int:
        saved_count = 0
        for content, result in zip(contents, results):
            if result:  
                self.save(content, result)
                saved_count += 1
        return saved_count
    
    def get_stats(self) -> Dict[str, Any]:
        if not os.path.exists(self.method_dir):
            return {"count": 0, "size_mb": 0.0, "valid_count": 0}
        
        files = [f for f in os.listdir(self.method_dir) if f.endswith('.json')]
        total_size = sum(
            os.path.getsize(os.path.join(self.method_dir, f))
            for f in files
        )
        
        
        valid_count = 0
        for f in files:
            
            if f.startswith(self.py_hash + "_"):
                valid_count += 1
        
        return {
            "total_count": len(files),
            "valid_count": valid_count,  
            "size_mb": total_size / (1024 * 1024),
            "method": self.method_name,
            "py_hash": self.py_hash,
            "path": self.method_dir
        }
    
    def clear_invalid(self) -> int:
        if not os.path.exists(self.method_dir):
            return 0
        
        removed = 0
        for f in os.listdir(self.method_dir):
            if f.endswith('.json') and not f.startswith(self.py_hash + "_"):
                os.remove(os.path.join(self.method_dir, f))
                removed += 1
        
        return removed
    
    def clear_all(self) -> int:
        import shutil
        if os.path.exists(self.method_dir):
            count = len([f for f in os.listdir(self.method_dir) if f.endswith('.json')])
            shutil.rmtree(self.method_dir)
            os.makedirs(self.method_dir, exist_ok=True)
            return count
        return 0


def get_all_cache_stats(cache_dir: str = None) -> Dict[str, Dict]:
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    
    if not os.path.exists(cache_dir):
        return {}
    
    stats = {}
    for method_name in os.listdir(cache_dir):
        method_dir = os.path.join(cache_dir, method_name)
        if os.path.isdir(method_dir):
            manager = KGCacheManager(method_name, cache_dir=cache_dir, auto_create=False)
            stats[method_name] = manager.get_stats()
    
    return stats


# ================================================================================

# ================================================================================


RETRY_SUFFIXES = ["", "2nd", "3rd", "4th", "5th"]


class BaseKGMethod(ABC):
    
    
    
    CHECK_LOG_DIR = os.path.join(EVAL_ROOT_DIR, "GeneratedKGContent", "_CheckLogs")
    
    def __init__(self, name: str = "BaseMethod", save_log: bool = True, **kwargs):
        self.name = name
        self.save_log = save_log
        self.config = kwargs
    
    def _save_check_log(self, input_content: str, raw_output: str, parsed_result: Dict[str, Any]) -> None:
        if not self.save_log:
            return
        
        
        method_dir = os.path.join(self.CHECK_LOG_DIR, self.name)
        os.makedirs(method_dir, exist_ok=True)
        
        
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        log_file = os.path.join(method_dir, f"{timestamp}.json")
        
        
        log_data = {
            "input": input_content,
            "output_raw": raw_output,
            "output_parsed": {
                "entities": parsed_result.get("entities", []),
                "relations": parsed_result.get("relations", [])
            },
            "timestamp": datetime.now().isoformat(),
            "method": self.name
        }
        
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        print(f"  📁 日志已保存: {log_file}")
        
    @abstractmethod
    def generate(self, content: str) -> Dict[str, Any]:
        raise NotImplementedError("子类必须实现 generate() 方法")
    
    def _robust_json_parse(self, response: str) -> Dict[str, Any]:
        import re
        
        if not response or not isinstance(response, str):
            return {'entities': [], 'relations': [], 'raw_output': str(response), 'error': 'Empty response'}
        
        
        json_str = response.strip()
        
        
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            
            entity_match = re.search(r'#Entity_List_Start#\s*(\[[\s\S]*?\])\s*#Entity_List_End#', response)
            relation_match = re.search(r'#Relationship_List_Start#\s*(\[[\s\S]*?\])\s*#Relationship_List_End#', response)
            
            if entity_match or relation_match:
                
                entities = []
                relations = []
                
                try:
                    if entity_match:
                        import json_repair
                        entities = json_repair.loads(entity_match.group(1))
                        
                except Exception as e:
                    
                    pass
                    
                try:
                    if relation_match:
                        import json_repair
                        relations = json_repair.loads(relation_match.group(1))
                        
                except Exception as e:
                    
                    pass
                    
                return {
                    'entities': entities,
                    'relations': relations,
                    'raw_output': response,
                    'parsed_method': 'split_list_regex'
                }

            
            brace_match = re.search(r'(\{[\s\S]*\})', response)
            if brace_match:
                json_str = brace_match.group(1).strip()
            else:
                
                bracket_match = re.search(r'(\[[\s\S]*\])', response)
                if bracket_match:
                    json_str = bracket_match.group(1).strip()
        
        
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                
                return self._distribute_list_data(data, response)
            if not isinstance(data, dict):
                raise ValueError(f"Parsed JSON is {type(data)}, expected dict")
            return {
                'entities': data.get('entities', []),
                'relations': data.get('relations', []),
                'raw_output': response
            }
        except (json.JSONDecodeError, ValueError):
            
            try:
                import json_repair
                data = json_repair.loads(json_str)
                if isinstance(data, list):
                    return self._distribute_list_data(data, response, repaired=True)
                if isinstance(data, dict):
                    
                    return {
                        'entities': data.get('entities', []),
                        'relations': data.get('relations', []),
                        'raw_output': response,
                        'repaired': True
                    }
            except Exception as e:
                
                pass
            
            
            repaired_str = self._repair_json(json_str)
            try:
                data = json.loads(repaired_str)
                if isinstance(data, list):
                    return self._distribute_list_data(data, response, repaired=True)
                if not isinstance(data, dict):
                    raise ValueError(f"Repaired JSON is {type(data)}, expected dict")
                
                return {
                    'entities': data.get('entities', []),
                    'relations': data.get('relations', []),
                    'raw_output': response,
                    'repaired': True
                }
            except Exception as e:
                
                return {
                    'entities': [],
                    'relations': [],
                    'raw_output': response,
                    'error': f'JSON parse failed even after repair: {e}'
                }

    def _distribute_list_data(self, data_list: List[Any], raw_output: str, repaired: bool = False) -> Dict[str, Any]:
        entities = []
        relations = []
        
        for item in data_list:
            if not isinstance(item, dict): continue
            
            
            is_rel = any(k in item for k in ['sub', 'obj', 'subject', 'object', 's', 'o', 'rel', 'relation', 'r'])
            if is_rel:
                relations.append(item)
            else:
                
                entities.append(item)
                
        return {
            'entities': entities,
            'relations': relations,
            'raw_output': raw_output,
            'parsed_from_list': True,
            'repaired': repaired
        }

    def _repair_json(self, json_str: str) -> str:
        json_str = json_str.strip()
        if not json_str:
            return "{}"
        
        
        quotes_count = json_str.count('"')
        if quotes_count % 2 != 0:
            json_str += '"'
        
        
        stack = []
        for char in json_str:
            if char == '{':
                stack.append('}')
            elif char == '[':
                stack.append(']')
            elif char == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
        
        
        repaired = json_str + "".join(reversed(stack))
        return repaired
    
    def batch_generate(self, contents: List[str], **kwargs) -> List[Dict[str, Any]]:
        results = []
        for i, content in enumerate(contents):
            print(f"  📝 [{i+1}/{len(contents)}] 正在生成...")
            result = self.generate(content)
            results.append(result)
        return results
    
    def format_output(self, result: Dict[str, Any]) -> str:
        entities = result.get('entities', [])
        relations = result.get('relations', [])
        
        
        
        return (
            f"#Final_Entity_List_Start#\n{entity_json}\n#Final_Entity_List_End#\n\n"
            f"#Final_Relationship_List_Start#\n{relation_json}\n#Final_Relationship_List_End#"
        )
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class LLMSinglePromptEndToEnd(BaseKGMethod):
    
    def __init__(
        self,
        name: str = "LLMSinglePrompt(Unnamed)",
        token: int = 17 * 1024,
        check_cache: bool = True,
        save_debug: bool = True,
        temp: float = 0.0,
        runtime_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.token = token
        self.temp = temp
        self.check_cache = check_cache
        self.runtime_context = runtime_context or {}
        self._grid_prompt_metadata: Optional[List[Dict[str, Any]]] = None

    def _build_prompt_metadata(
        self,
        contents: List[str],
        *,
        attempt: int,
        stage: str = "generate",
    ) -> List[Dict[str, Any]]:
        prompt_metadata: List[Dict[str, Any]] = []
        for content in contents:
            sample_ref = lookup_sample_ref(self.runtime_context, content)
            metadata = dict(sample_ref)
            metadata["stage"] = stage
            metadata["attempt"] = attempt
            prompt_metadata.append(metadata)
        return prompt_metadata
    
    def _create_prompt(self, content: str) -> List[Dict]:
        
        prompt_builder = getattr(
            local_tools_prompt,
            "grid_kg_single_prompt_maker_very_simple_20260303",
            getattr(local_tools_prompt, "grid_kg_single_prompt_maker_very_simple", None),
        )
        if prompt_builder is None:
            raise ValueError("Local prompt bundle is missing grid_kg_single_prompt_maker_very_simple_20260303 and its legacy fallback.")
        p_data = prompt_builder(content)
        
        
        
        
        base_prompt = []
        if isinstance(p_data, list) and p_data:
            base_prompt = p_data
        elif isinstance(p_data, str):
            base_prompt = [{"role": "user", "content": p_data}]
        else:
            raise ValueError(f"local prompt builder returned an unexpected type: {type(p_data)}")
            
        
        
        return [msg.copy() for msg in base_prompt]
    
    def _create_retry_prompt(self, base_prompt: List[Dict], attempt: int) -> List[Dict]:
        current_prompt = [msg.copy() for msg in base_prompt]
        
        if attempt > 0:
            suffix = RETRY_SUFFIXES[min(attempt, len(RETRY_SUFFIXES) - 1)]
            suffix_text = (
                f"\n\nYour last response was incomplete or invalid JSON. "
                f"Please output a valid JSON object inside a code block. "
                f"This is {suffix} attempt."
            )
            current_prompt[-1]['content'] += suffix_text
        
        return current_prompt

    def _check_completeness(self, response: str) -> bool:
        return response and '<Fin>' in response
    
    def _clean_response(self, response: str) -> str:
        if response:
            return response.replace('<Fin>', '').strip()
        return ''
    
    @abstractmethod
    def _call_llm(self, prompt_list: List[List[Dict]]) -> List[str]:
        raise NotImplementedError("子类必须实现 _call_llm() 方法")

    def generate(self, content: str, max_retries: int = 3) -> Dict[str, Any]:
        base_prompt = self._create_prompt(content)
        try:
            self._grid_prompt_metadata = self._build_prompt_metadata([content], attempt=0, stage="generate")
            responses = self._call_llm([base_prompt])
            response = responses[0] if responses else ''
            self._grid_prompt_metadata = None
            clean_response = self._clean_response(response)
            result = self._robust_json_parse(clean_response)
            self._save_check_log(content, response, result)
            return result
        except Exception as e:
            self._grid_prompt_metadata = None
            print(f"  ❌ LLM 调用失败: {e}")
            result = {
                'entities': [],
                'relations': [],
                'raw_output': '',
                'error': str(e)
            }
            self._save_check_log(content, '', result)
            return result
    
    def batch_generate(self, contents: List[str], max_retries: int = 3, **kwargs) -> List[Dict[str, Any]]:
        print(f"🚀 批量生成 {len(contents)} 篇文章的知识图谱...")
        
        prompt_list = [self._create_prompt(c) for c in contents]
        print(f"🚀 单轮发送 {len(prompt_list)} 个请求...")
        self._grid_prompt_metadata = self._build_prompt_metadata(
            contents,
            attempt=0,
            stage="generate",
        )
        try:
            responses = self._call_llm(prompt_list)
        finally:
            self._grid_prompt_metadata = None

        results = [None] * len(contents)
        for idx, resp in enumerate(responses):
            clean_resp = self._clean_response(resp)
            results[idx] = self._robust_json_parse(clean_resp)

        for idx in range(len(results)):
            if results[idx] is None:
                results[idx] = {'entities': [], 'relations': [], 'error': 'Empty response'}

        success_count = sum(1 for r in results if r.get('relations'))
        print(f"✅ 批量生成完成: {success_count}/{len(contents)} 成功")
        
        return results


# ================================================================================

# ================================================================================

if __name__ == "__main__":
    print("🗂️ 知识图谱核心模块 (缓存管理 + 输入输出)")
    print("=" * 60)
    print("功能 1: 缓存管理 - py文件hash + 输入string hash = 缓存key")
    print("功能 2: 基类 - BaseKGMethod, LLMSinglePromptEndToEnd")
    print("=" * 60)
    print()
    
    
    print("📊 当前缓存统计:")
    all_stats = get_all_cache_stats()
    
    if not all_stats:
        print("   (无缓存)")
    else:
        total_count = 0
        total_valid = 0
        total_size = 0.0
        for method, stats in all_stats.items():
            print(f"   📁 {method}:")
            print(f"      总计: {stats['total_count']} 条, 有效: {stats['valid_count']} 条, {stats['size_mb']:.2f} MB")
            print(f"      py_hash: {stats['py_hash']}")
            total_count += stats['total_count']
            total_valid += stats['valid_count']
            total_size += stats['size_mb']
        print(f"   ─────────────────────────")
        print(f"   📊 合计: {total_count} 条 (有效 {total_valid}), {total_size:.2f} MB")
