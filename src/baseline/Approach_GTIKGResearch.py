# -*- coding: utf-8 -*-

import os
import sys
import json
import re
import importlib.util
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from datetime import datetime


DROPBOX_PATH = os.path.join(os.path.expanduser("~"), 'Dropbox')
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


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


# ================================================================================

# ================================================================================


EXTRACTION_PROMPT = """As an AI trained in entity extraction and relationship extraction. You're an advanced AI expert, so even if I give you a complex sentence, you'll still be able to perform the relationship extraction task. The output format MUST be a dictionary where key is the source sentence and value is a list consisting of the extracted triple.

A triple is a basic data structure used to represent knowledge graphs, which are structured semantic knowledge bases that describe concepts and their relationships in the physical world. A triple MUST has THREE elements: [Subject, Relation, Object]. The subject and the object are Noun. The relation is a relation that connects the subject and the object, and expresses how they are related.

In entity extraction, you follow those rules:
Rule 1: Only extract triples that are related to cyber attacks. If a sentence does not have any triple about cyber attacks, skip the sentence and do not print it in your output.
Rule 2: Make sure your results is a python dictionary format. One example is {{source sentence1:[[subject1, relation1, object1],[subject2, relation2, object2]...],source sentence2:[[subject3, relation3, object3],[subject4, relation4, object4]...]}} 
Rule 3: You must use ellipsis in source sentence to save space. The output format should be "First word Second word ... penu word last word", For example, "The malware ... the system".

Example:
Input: "Leafminer attempts to infiltrate target networks through various means of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts."
Output: {{Leafminer attempts ... of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts:[[SUBJECT:Leafminer,RELATION:attempts to infiltrate,OBJECT:target networks],[SUBJECT:Leafminer,RELATION:use,OBJECT:watering hole websites],[SUBJECT:Leafminer,RELATION:use,OBJECT:vulnerability scans of network services on the internet],[SUBJECT:Leafminer,RELATION:use,OBJECT:brute-force],[SUBJECT:Leafminer,RELATION:use,OBJECT:dictionary login attempts]]}}.

Now extract all possible entity triples from this text:
---
{content}
---

You MUST follow the rules I told you before. Output ONLY the dictionary."""



MERGE_3_RESULTS_PROMPT = """You are responsible for combining the three different entity extraction results from three different assistants extracting from the same sentence into one. The triple is a basic data structure used to represent knowledge graphs, which are structured semantic knowledge bases that describe concepts and their relationships in the world. A triple MUST have THREE elements: [subject, relation, object]. The subject has the prefix "SUBJECT:",the relation has prefix "RELATION:", the object has prefix "OBJECT:", some triples examples are "[SUBJECT:The user, RELATION:logs in, OBJECT:the system],[SUBJECT:The system, RELATION:stores, personal information],[SUBJECT:The system, RELATION:sends, OBJECT:personal information]". The final results is a python dictionary format. One example of result is {{source sentence1:[[subject1, relation1, object1],[subject2, relation2, object2]...]}}. Some assistants use ellipses to simplify words source sentence, for example "The exploit was delivered through a Microsoft Office document and the final payload was the latest version of FinSpy malware." and "The exploit ... FinSpy malware." and "The exploit was delivered ... latest version of FinSpy malware." are the same one sentence. So when you find the different dictionary key that has same beginning and ending words, you should combine them into one dict. I would like you to integrate these three results into one and discard the exact same triples and discard triples that do not contain exactly 3 elements.

The source sentence is: {source_sentence}
The extracted triples results are: {results_list}

Just answer me the final python dictionary with triple format without any other words."""



POSTPROCESS_PROMPT = """You play the role of an entity extraction expert and modify/simplify/split the text (extracted multiple triples) in the entity extraction result I gave you (a python dictionary with key as the source sentence with ellipsis and value as the extracted triples) according to the following rules. A triple is a basic data structure used to represent knowledge graphs, which are structured semantic knowledge bases that describe concepts and their relationships in the physical world. A triple consists of three elements: [SUBJECT, RELATION,OBJECT]. The subject and the object are entities, which can be things, people, places, events, or abstract concepts. The relation is a relation that connects the subject and the object, and expresses how they are related.

Rule 1: If the subject or object in a triple contains pronouns such as it, they, malware, Trojan, attack, ransomware, or group, replace them with a specific name as much as possible according to the context.
Rule 2: Focus on malware, Trojan horse, CVE, or hacking organization as the subject of the triples, if a subject with "malware" or "Trojan horse" or "CVE" or "hacking organization" is found and has additional suffixes, remove the suffixes.
Rule 3: Split a complex triple into multiple simpler forms. For example, [Formbook and XLoader, are,malware] should be split into [Formbook,is,malware] and [XLoader,is,malware].
Rule 4: If the [subject,relation] in a triple can be formed into a new [subject,relation,object] triple because relation itself has a new object in it, create a new triple while keeping the original one. 
Rule 5: If the object can be simplified to a more concise, generic expression, create a new triple while keeping the original one.
Rule 6: Simplify the subject, object, and relation into a more concise, generic expression.
Rule 7: When you encounter a subject or object that contains modifiers and adjectives, remove them. For example, [a notorious Formbook malware] should be simplified to [Formbook].
Rule 8: When you encounter a plural or past tense form, convert it to singular or present tense. For example, [Windows users] should be converted to [Windows user].
Rule 9: When you encounter an MD5, registry, path, or other identifier that contains prefixes, remove them. For example, [md5 xxxxx] should be simplified to [xxxxx].
Rule 10: When you encounter a proper noun that contains a suffix, remove the suffix. For example, ["Specific names of a malware/ransomware/trojan" malware/ransomware/trojan] should be simplified to ["Specific names of a malware/ransomware/trojan"]
Rule 11: Make sure the subject has a prefix "SUBJECT:", the relation has prefix "RELATION:", the object has prefix "OBJECT:".

Here is my entity extraction result: {triples}

Now, you apply the rules I told you before. Write down your thought, think it step by step. If all triple don't need to be modified based on specific rule, just write down 'no change'. In the end, you MUST tell me the final new entity extraction result. Make sure your results contain a dictionary where key is the original sentence and value is a list consisting of the extracted triple for subsequent information extraction."""



MERGE_MEMORY_PROMPT = """You are a triples integration assistant. Triple is a basic data structure, which describes concepts and their relationships. A triple in long-term and short-term memory MUST has THREE elements: [Subject, Relation, Object]. You are now reading a whole article and extract all triples from it. But you can only see part of the article at a time. In order to record all the triples from a article, you have the following long-term memory area to record the triples from the entire article. long-term memory stores information on the aricle parts you have already read.

-The start of the long-term memory area-
{longmem}
-The end of the long-term memory area-

Second, you now see a part of this article. Based on this part, you already extract such triples and place them in your short-term memory: 

-The start of the short-term memory area-
{shortmem}
-The end of the short-term memory area-

Third, now review your long-term memory and short-term memory. Modify the short-term memory into a new short-term memory. You should follow following rules to modify triples in short-term memory to make them consistent with triples in long-term memory:

Rule 1. You notice that in these triples, some triples have subjects and objects that contain partially identical terms and refer to the same specific nouns, but these specific nouns have prefixes/suffixes/modifiers that make them not identical. You should delete the prefixes/suffixes/modifiers and unify them into the same specific nouns.

Rule 2. Be especially careful that when you meet specific names of malware, CVE, Trojans, hacker organizations, etc., always use their specific names and remove the prefixes/suffixes/modifiers.

Rule 3. Don't add unexisting triples to your new short-term memory. 

Rule 4. Don't add unexisting triples that don't exist in long-term memory or short-term memory to your new short-term memory. You should add triples from long-term memory or short-term memory to your new short-term memory, not from your imagination and selfcreation.

Rule 5. Don't add any example word like 'Formbook','XLoader','Leafminer', 'FinSpy', 'Kismet' in your new short-term memory area, they are just example words not the real triples in the long term memory area or short term memory area.

Rule 6. new short-term memory area must be started with '-The start of new short-term memory area-' and ended with '-The end of new short-term memory area-'. A triple in new short-term memory MUST has THREE elements: [Subject, Relation, Object].

Now, follow the rules. Write down how you use the rule to modify the triples in short-term memory. Then, write down new short-term memory which must be started with '-The start of new short-term memory area-' and ended with '-The end of new short-term memory area-'"""


# ================================================================================

# ================================================================================


EXAMPLE_KEYWORDS = ['Formbook', 'XLoader', 'Leafminer', 'FinSpy', 'Kismet', 
                    'Agumon', 'Gabumon', 'Biyomon', '2042', 'CVExxx', 
                    'Malwaresavetextfile', 'savetextfile', 'Specificnamesofa']


def clean_text_for_check(text: str) -> str:
    import string
    if not isinstance(text, str):
        return str(text)
    cleaned_text = re.sub(r'[^\x20-\x7E]', '', text)
    cleaned_text = re.sub(r'[\s{}]+'.format(re.escape(string.punctuation)), '', cleaned_text)
    cleaned_text = re.sub(r'SUBJECT|RELATION|OBJECT', '', cleaned_text)
    return cleaned_text if cleaned_text else 'Null'


def get_only_triples(text: str) -> str:
    text = text.replace(': [', ':[') 
    if "{" in text and "}" in text:
        start_index = text.rindex('{')
        end_index = text.rindex('}') + 1
        triple_only_text = text[start_index:end_index]
        if ":[" in triple_only_text and ']' in triple_only_text:
            source_sentence = triple_only_text.split(':')[0] 
            source_sentence = source_sentence.split('{')[1]
            words = source_sentence.split() 
            if len(words) <= 2:
                abbreviation = source_sentence
            else:
                abbreviation = " ".join(words[:2]) + " ... " + " ".join(words[-2:])
            triple_only_text = triple_only_text.replace(source_sentence, abbreviation)     
    else:
        text = text.replace('\n', '')
        if ":[" in text and "]" in text:
            start_index = text.rindex(':[')
            end_index = text.rindex(']') + 1
            triple_only_text = text[start_index:end_index]
        else:
            if "[[" in text and "]]" in text:
                start_index = text.rindex('[[')
                end_index = text.rindex(']]') + 1
                triple_only_text = text[start_index:end_index]
            else:
                triple_only_text = text
    return triple_only_text


def clean_full_extracted_triples(text: str) -> str:
    text = text.replace('\n', '')
    text = re.sub(r':\s+\[', r'[', text)
    text = re.sub(r'\s+\[', r'[', text)
    text = re.sub(r'\s+\]', r']', text)
    text = re.sub(r'\]\s*,\s*\]', ']]', text)
    
    triple_only_text = text
    if "[[" in text and "]]" in text:
        start_index = text.rindex('[[') + 1
        end_index = text.rindex(']]') + 1
        triple_only_text = text[start_index:end_index]
    else:
        if "[[" in text or "]]" in text:
            start_index = text.index('[')
            end_index = text.rindex(']') + 1
            triple_only_text = text[start_index:end_index]
    
    triple_only_text = triple_only_text.replace('"', '')
    triple_only_text = triple_only_text.replace("'", '')
    return triple_only_text


def check_brackets(my_string: str) -> bool:
    if my_string is None or len(my_string) == 0:
        return False
    my_string = my_string.strip()
    return my_string[0] == '[' and my_string[-1] == ']'


def contains_example_keywords(text: str) -> bool:
    cleaned = clean_text_for_check(text)
    return any(kw in cleaned for kw in EXAMPLE_KEYWORDS)


def full_text_to_parts(text: str, max_chunk_size: int = 600, min_chunk_size: int = 20) -> List[str]:
    
    paragraphs = text.split('\n')
    
    processed_paragraphs = []
    
    for paragraph in paragraphs:
        if len(paragraph) > max_chunk_size:
            
            try:
                sentences = nltk.sent_tokenize(paragraph)
            except:
                sentences = [paragraph]
            
            
            sentences = [x[:500] for x in sentences]
            
            new_paragraph = ''
            for sentence in sentences:
                temp_length = len(new_paragraph) + len(sentence)
                if temp_length < max_chunk_size:
                    new_paragraph += (sentence + '\n')
                else:
                    if len(new_paragraph) >= min_chunk_size:
                        processed_paragraphs.append(new_paragraph.strip())
                    new_paragraph = sentence + '\n'
            
            if len(new_paragraph) >= min_chunk_size:
                processed_paragraphs.append(new_paragraph.strip())
        else:
            if len(paragraph) >= min_chunk_size:
                processed_paragraphs.append(paragraph.strip())
    
    
    combined_paragraphs = []
    current_combined = ''
    
    for paragraph in processed_paragraphs:
        temp_length = len(current_combined) + len(paragraph) + 1
        if temp_length < max_chunk_size:
            current_combined += (' ' if current_combined else '') + paragraph
        else:
            if current_combined:
                combined_paragraphs.append(current_combined)
            current_combined = paragraph
    
    if current_combined:
        combined_paragraphs.append(current_combined)
    
    return combined_paragraphs


# ================================================================================

# ================================================================================

class GTIKGResearchMethod:
    
    
    
    CHECK_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GeneratedKGContent", "_CheckLogs")
    
    def __init__(
        self,
        model: str = 'gpt-5-nano',
        token: int = 17 * 1024,
        temperatures: List[float] = [1.0, 0.5, 0.2],
        max_chunk_size: int = 600,
        max_retries: int = 3,
        save_log: bool = True,
        use_cloud_or_vllm: str = 'cloud',
        vllm_model_path: str = None,
        quick_merger: bool = False,  
        max_batch_workers: int = 8,
        shared_llm_backend: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.shared_llm_backend = dict(shared_llm_backend or {})
        self.runtime_context = runtime_context or {}
        self.use_cloud_or_vllm = 'vllm' if self.shared_llm_backend.get('enabled') else use_cloud_or_vllm.lower()
        self.vllm_model_path = self.shared_llm_backend.get('model_path', vllm_model_path)
        self.model = self.shared_llm_backend.get('model', (model if self.use_cloud_or_vllm == 'cloud' else 'local'))
        self.token = token
        self.temperatures = temperatures
        self.max_chunk_size = max_chunk_size
        self.max_retries = 1
        self.name = "GTIKGResearch"
        self.save_log = save_log
        self.quick_merger = quick_merger  
        self.max_batch_workers = max(1, int(max_batch_workers or 1))
        self.ask_check_history_cache = self.shared_llm_backend.get('check_history_cache', True)
        self.ask_vllm_smart_mode = self.shared_llm_backend.get('smart_mode', True)
        self.ask_max_workers_vllm = self.shared_llm_backend.get('max_workers_vllm', 'auto')
        self.ask_prompt_send_weight_vllm = self.shared_llm_backend.get(
            'prompt_send_weight_vllm',
            {"super": 2, "ultra": 4, "normal": 1},
        )
        self.ask_vllm_server_name = self.shared_llm_backend.get('vllm_server_name')
        self._grid_current_stage = "extract"
        self._grid_current_sample_ref = {}
        
        
        self.vllm_manager = None
        if self.use_cloud_or_vllm == 'vllm' and not self.shared_llm_backend.get('enabled'):
            if not vllm_model_path:
                raise ValueError("❌ 使用 vLLM 模式时必须指定 vllm_model_path 参数")
            from vllm_environment_setup import VLLMEnvironmentManager
            self.vllm_manager = VLLMEnvironmentManager(vllm_model_path)
            print(f"✅ {self.name} 初始化 (🖥️ vLLM 模式: {vllm_model_path})")
        elif self.use_cloud_or_vllm == 'vllm':
            print(f"✅ {self.name} 初始化 (🖥️ 共享三机 vLLM 模式: {self.vllm_model_path})")
        else:
            mode_info = "⚡快速合并" if quick_merger else "完整流程"
            print(f"✅ {self.name} 初始化 (☁️ 云端模式: {model}, temps={temperatures}, {mode_info})")
        if int(max_retries or 1) != 1:
            print(f"⚠️ [{self.name}] 新目录固定关闭重试，收到 max_retries={max_retries} 仅作兼容处理")
        
        if KGCacheManager:
            self.cache = KGCacheManager(os.path.basename(__file__), method_file_path=os.path.abspath(__file__))
        else:
            self.cache = None
    
    def _save_check_log(self, input_content: str, raw_output: str, parsed_result: dict) -> None:
        if not self.save_log:
            return
        method_dir = os.path.join(self.CHECK_LOG_DIR, self.name)
        os.makedirs(method_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        log_file = os.path.join(method_dir, f"{timestamp}.json")
        log_data = {
            "input": input_content,
            "output_raw": raw_output,
            "output_parsed": {"relations": parsed_result.get("relations", [])},
            "timestamp": datetime.now().isoformat(),
            "method": self.name
        }
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        print(f"  📁 日志已保存: {log_file}")
    
    def _call_llm(self, prompt: str, temp: float = 0.5) -> str:
        
        if self.vllm_manager:
            self.vllm_manager.ensure_ready()
        messages = [{"role": "user", "content": prompt}]
        results = run_logged_asks(
            [messages],
            model=self.model,
            token=self.token,
            temp=temp,
            runtime_context=self.runtime_context,
            phase="generate",
            prompt_metadata_list=[dict(self._grid_current_sample_ref, stage=self._grid_current_stage)],
            check_history_cache=self.ask_check_history_cache,
            VllmSmartMode=self.ask_vllm_smart_mode,
            max_workers_Vllm=self.ask_max_workers_vllm,
            prompt_send_weight_VllmNotSmartMode=self.ask_prompt_send_weight_vllm,
            vllm_server_name=self.ask_vllm_server_name,
        )
        return results[0] if results else ""
    
    def _extract_single_chunk(self, chunk_text: str) -> str:
        
        if len(chunk_text) > 1500:
            chunk_text = chunk_text[:1500]
        
        
        if self.quick_merger:
            temp = self.temperatures[0] if self.temperatures else 0.5
            prompt = EXTRACTION_PROMPT.format(content=chunk_text)
            self._grid_current_stage = "chunk_extract_quick"
            result = self._call_llm(prompt, temp=temp)
            if contains_example_keywords(result):
                return ""
            return get_only_triples(result)
        
        
        
        first_answer_list = []
        for temp in self.temperatures:
            prompt = EXTRACTION_PROMPT.format(content=chunk_text)
            self._grid_current_stage = f"chunk_extract_t{temp}"
            result = self._call_llm(prompt, temp=temp)
            
            
            if contains_example_keywords(result):
                first_answer_list.append('ERROR')
            else:
                first_answer_list.append(get_only_triples(result))
        
        
        valid_results = [r for r in first_answer_list if r != 'ERROR' and r.strip()]
        if not valid_results:
            return ""
        
        if len(valid_results) == 1:
            merged = valid_results[0]
        else:
            prompt = MERGE_3_RESULTS_PROMPT.format(
                source_sentence=chunk_text[:200] + "...",
                results_list=str(valid_results[:3])
            )
            self._grid_current_stage = "chunk_merge"
            merged = self._call_llm(prompt, temp=0.5)
            merged = get_only_triples(merged)
        
        
        prompt = POSTPROCESS_PROMPT.format(triples=merged)
        self._grid_current_stage = "chunk_postprocess"
        postprocessed = self._call_llm(prompt, temp=0.7)
        extracted_text = get_only_triples(postprocessed)
        
        return extracted_text
    
    def _merge_memory(self, longmem: str, shortmem: str, source_text: str) -> str:
        
        longmem_for_prompt = longmem
        if len(longmem) >= 1500:
            longmem_for_prompt = longmem[-1000:]
            if '[' in longmem_for_prompt:
                longmem_for_prompt = longmem_for_prompt[longmem_for_prompt.index('['):]
        
        prompt = MERGE_MEMORY_PROMPT.format(
            longmem=longmem_for_prompt,
            shortmem=shortmem
        )
        
        self._grid_current_stage = "memory_merge"
        result = self._call_llm(prompt, temp=0.7)
        
        
        result = result.replace('-The start of the new short-term memory area-', '-The start of new short-term memory area-')
        result = result.replace('-The end of the new short-term memory area-', '-The end of new short-term memory area-')
        
        
        if '-The start of new short-term memory area-' in result and '-The end of new short-term memory area-' in result:
            start_marker = '-The start of new short-term memory area-'
            end_marker = '-The end of new short-term memory area-'
            new_shortmem = result[result.rindex(start_marker) + len(start_marker):result.rindex(end_marker)]
            
            
            if not contains_example_keywords(new_shortmem):
                
                return longmem + ', ' + new_shortmem.strip()
        
        
        return longmem + ', ' + shortmem
    
    def _parse_to_json(self, raw_output: str) -> Dict[str, Any]:
        entities = []
        relations = []
        
        
        pattern = r'\[SUBJECT:([^,\]]+),\s*RELATION:([^,\]]+),\s*OBJECT:([^\]]+)\]'
        matches = re.findall(pattern, raw_output, re.IGNORECASE)
        
        entity_set = set()
        for sub, rel, obj in matches:
            sub = sub.strip()
            obj = obj.strip()
            rel = rel.strip()
            
            
            if sub not in entity_set:
                entities.append({"name": sub, "type": "CTI_Entity"})
                entity_set.add(sub)
            if obj not in entity_set:
                entities.append({"name": obj, "type": "CTI_Entity"})
                entity_set.add(obj)
            
            
            relations.append({"sub": sub, "rel": rel, "obj": obj})
        
        
        return {
            "relations": relations,
            "raw_output": raw_output
        }
    
    def generate(self, content: str) -> str:
        print(f"🚀 [{self.name}] 开始处理 (完整复现模式)...")
        self._grid_current_sample_ref = lookup_sample_ref(self.runtime_context, content)

        if self.cache:
            cached_result = self.cache.load(content)
            if cached_result is not None:
                print(f"✨ [{self.name}] 命中缓存: chars={len(content)}")
                self._grid_current_sample_ref = {}
                self._grid_current_stage = "extract"
                return cached_result
        
        
        print(f"  📝 阶段 1: 文章分段...")
        chunks = full_text_to_parts(content, max_chunk_size=self.max_chunk_size)
        print(f"  📊 文章已拆分为 {len(chunks)} 个段落")
        
        if not chunks:
            print("  ⚠️ 文章内容过短或无法分段")
            self._grid_current_sample_ref = {}
            return json.dumps({"relations": [], "raw_output": ""}, ensure_ascii=False)
        
        
        longmem = ""
        
        for i, chunk in enumerate(chunks):
            print(f"  🔄 阶段 2: 处理段落 {i+1}/{len(chunks)}...")
            
            
            shortmem = None
            extracted = self._extract_single_chunk(chunk)
            cleaned = clean_full_extracted_triples(extracted)
            
            
            if not contains_example_keywords(cleaned) and check_brackets(cleaned):
                shortmem = cleaned
            else:
                print(f"    ⚠️ 段落 {i+1} 提取结果无效，按无重试策略直接跳过")
            
            if shortmem is None:
                print(f"    ❌ 段落 {i+1} 提取失败，跳过")
                continue
            
            print(f"    ✅ 段落 {i+1} 短期记忆: {shortmem[:100]}...")
            
            
            if self.quick_merger:
                
                if longmem:
                    longmem = longmem + ', ' + shortmem
                else:
                    longmem = shortmem
                print(f"    ⚡ 快速拼接到结果")
            elif i == 0:
                
                longmem = shortmem
                print(f"    🧠 初始化长期记忆")
            else:
                
                print(f"    🧠 合并到长期记忆...")
                longmem = self._merge_memory(longmem, shortmem, chunk)
        
        print(f"  ✅ 阶段 2 完成: 所有段落已处理")
        
        
        print(f"  📊 阶段 3: 解析最终结果...")
        result = self._parse_to_json(longmem)
        
        print(f"  ✅ 完成: {len(result['relations'])} 关系")
        
        
        self._save_check_log(content, longmem, result)
        final_output = json.dumps(result, ensure_ascii=False)
        if self.cache:
            self.cache.save(content, final_output)
        
        
        self._grid_current_sample_ref = {}
        self._grid_current_stage = "extract"
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

        def _run_single(idx: int, content: str):
            try:
                return idx, json.loads(self.generate(content))
            except Exception as exc:
                print(f"  ⚠️ [{self.name}] 样本 {idx + 1} 失败: {exc}")
                return idx, {"relations": [], "raw_output": str(exc), "error": str(exc)}

        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="gtikg_batch") as executor:
            future_map = {executor.submit(_run_single, idx, content): idx for idx, content in enumerate(contents)}
            completed = 0
            for future in as_completed(future_map):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                if completed % 5 == 0 or completed == len(contents):
                    print(f"  🔄 [{self.name}] batch 进度: {completed}/{len(contents)}")

        return [item if item is not None else {"relations": [], "raw_output": "", "error": "Unknown gap"} for item in results]



Method = GTIKGResearchMethod


if __name__ == "__main__":
    print("🚀 GTIKGResearch 知识图谱生成方法 (完整复现版)")
    print("使用: from 方法_GTIKGResearch import Method")
    
    
    test_text = """APT29 deployed the SUNBURST malware targeting SolarWinds Orion software. The malware used DNS tunneling for command and control communication.

The attackers first gained access through a supply chain compromise. They modified the Orion software build process to inject malicious code.

After initial access, APT29 moved laterally within victim networks. They used various techniques including credential dumping and pass-the-hash attacks."""
    
    method = Method()
    result = method.generate(test_text)
    print("\n📊 测试结果:")
    print(result)
