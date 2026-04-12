# -*- coding: utf-8 -*-

import logging
import json
import re
import importlib.util
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys


DROPBOX_PATH = os.path.join(os.path.expanduser("~"), 'Dropbox')
if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)

# Ensure current directory is in path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

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

# ================================================================================

# ================================================================================

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
 
-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)
 
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)
 
3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.
 
4. When finished, output <|COMPLETE|>
 
######################
-Examples-
######################
Example 1:
Entity_types: ORGANIZATION,PERSON
Text:
The Verdantis's Central Institution is scheduled to meet on Monday and Thursday, with the institution planning to release its latest policy decision on Thursday at 1:30 p.m. PDT, followed by a press conference where Central Institution Chair Martin Smith will take questions. Investors expect the Market Strategy Committee to hold its benchmark interest rate steady in a range of 3.5%-3.75%.
######################
Output:
("entity"<|>CENTRAL INSTITUTION<|>ORGANIZATION<|>The Central Institution is the Federal Reserve of Verdantis, which is setting interest rates on Monday and Thursday)
##
("entity"<|>MARTIN SMITH<|>PERSON<|>Martin Smith is the chair of the Central Institution)
##
("entity"<|>MARKET STRATEGY COMMITTEE<|>ORGANIZATION<|>The Central Institution committee makes key decisions about interest rates and the growth of Verdantis's money supply)
##
("relationship"<|>MARTIN SMITH<|>CENTRAL INSTITUTION<|>Martin Smith is the Chair of the Central Institution and will answer questions at a press conference<|>9)
<|COMPLETE|>

######################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:"""

CONTINUE_PROMPT = "MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:\n"
LOOP_PROMPT = "It appears some entities and relationships may have still been missed. Answer Y if there are still entities or relationships that need to be added, or N if there are none. Please answer with a single letter Y or N.\n"

DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event", "malware", "tool", "vulnerability", "attack_pattern", "campaign", "identity", "infrastructure", "location", "software", "threat_actor"]

class GraphRAGMethod:
    def __init__(self, model='gpt-5-nano', token=16*1024, temp=0.1, max_gleanings=1,
                 use_cloud_or_vllm='cloud', vllm_model_path=None, shared_llm_backend=None, runtime_context=None, **kwargs):
        self.shared_llm_backend = dict(shared_llm_backend or {})
        self.runtime_context = runtime_context or {}
        self.use_cloud_or_vllm = 'vllm' if self.shared_llm_backend.get('enabled') else use_cloud_or_vllm.lower()
        self.vllm_model_path = self.shared_llm_backend.get('model_path', vllm_model_path)
        self.model = self.shared_llm_backend.get('model', (model if self.use_cloud_or_vllm == 'cloud' else 'local'))
        self.token = token
        self.temp = temp
        self.max_gleanings = max_gleanings
        self.name = "GraphRAG"
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
            print(f"  🖥️ GraphRAG vLLM 模式: {vllm_model_path}")
        elif self.use_cloud_or_vllm == 'vllm':
            print(f"  🖥️ GraphRAG 共享三机 vLLM 模式: {self.vllm_model_path}")
        else:
            print(f"  ☁️ GraphRAG 云端模式: {model}")

        if KGCacheManager:
            self.cache = KGCacheManager(os.path.basename(__file__), method_file_path=os.path.abspath(__file__))
        else:
            self.cache = None

    def _call_llm(self, messages):
        
        if self.vllm_manager:
            self.vllm_manager.ensure_ready()
        prompt_metadata = [dict(self._grid_current_sample_ref, stage=self._grid_current_stage)]
        res = run_logged_asks(
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
        return res[0] if res else ""
    
    def generate(self, content: str) -> str:
        print(f"🚀 [{self.name}] Running...")
        self._grid_current_sample_ref = lookup_sample_ref(self.runtime_context, content)
        
        # 1. Check Cache
        if self.cache:
            cached_res = self.cache.load(content)
            if cached_res is not None:
                print(f"✨ Loaded result from cache for content ({len(content)} chars)")
                self._grid_current_sample_ref = {}
                return cached_res

        # Initial Extraction
        prompt = GRAPH_EXTRACTION_PROMPT.format(
            entity_types=",".join([t.upper() for t in DEFAULT_ENTITY_TYPES]),
            input_text=content
        )
        
        history = [{"role": "user", "content": prompt}]
        self._grid_current_stage = "extract"
        response = self._call_llm(history)
        history.append({"role": "assistant", "content": response})
        
        results = [response]
        
        # Loop (Gleaning)
        for i in range(self.max_gleanings):
             # Ask if missed
             check_history = history + [{"role": "user", "content": LOOP_PROMPT}]
             self._grid_current_stage = "loop_check"
             check_resp = self._call_llm(check_history)
             
             if check_resp.strip().upper() == 'Y':
                 print(f"  🔄 Gleaning iteration {i+1}...")
                 cont_history = history + [{"role": "user", "content": CONTINUE_PROMPT}]
                 self._grid_current_stage = "loop_continue"
                 cont_resp = self._call_llm(cont_history)
                 history.append({"role": "user", "content": CONTINUE_PROMPT})
                 history.append({"role": "assistant", "content": cont_resp})
                 results.append(cont_resp)
             else:
                 break

        # Parse results
        entities = []
        relations = []
        seen_ents = set()
        
        print(f"  Parsing {len(results)} extraction blocks...")
        
        for block in results:
            items = block.split("##")
            for item in items:
                item = item.strip()
                if item.startswith('("entity"'):
                    # ("entity"<|>NAME<|>TYPE<|>DESC)
                    parts = item[1:-1].split("<|>")
                    if len(parts) >= 4:
                        name = parts[1]
                        type_ = parts[2]
                        # desc = parts[3]
                        if name not in seen_ents:
                            entities.append({"name": name, "type": type_})
                            seen_ents.add(name)
                            
                elif item.startswith('("relationship"'):
                     # ("relationship"<|>SRC<|>TGT<|>DESC<|>SCORE)
                    parts = item[1:-1].split("<|>")
                    if len(parts) >= 5:
                        src = parts[1]
                        tgt = parts[2]
                        # desc = parts[3]
                        # score = parts[4]
                        relations.append({"sub": src, "rel": "related_to", "obj": tgt})
                        
                        # Ensure entities exist
                        if src not in seen_ents:
                            entities.append({"name": src, "type": "Unknown"})
                            seen_ents.add(src)
                        if tgt not in seen_ents:
                             entities.append({"name": tgt, "type": "Unknown"})
                             seen_ents.add(tgt)
        
        
        final_output = json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False)

        # 2. Save Cache
        if self.cache:
            self.cache.save(content, final_output)
        self._grid_current_sample_ref = {}
        self._grid_current_stage = "extract"
        
        return final_output

Method = GraphRAGMethod

if __name__ == "__main__":
    m = Method()
    print(m.generate("Microsoft warned about the Midnight Blizzard attack."))
