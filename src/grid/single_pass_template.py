# -*- coding: utf-8 -*-

import os
import sys
import json
import time
from typing import List, Dict, Any, Optional




if os.environ.get("GRID_FORCE_DISABLE_CUDA_VISIBLE_DEVICES") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


DROPBOX_PATH = os.path.join(os.path.expanduser("~"), 'Dropbox')
if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)


_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from article_io_cache_parser import LLMSinglePromptEndToEnd, RETRY_SUFFIXES
from vllm_environment_setup import VLLMEnvironmentManager, SERVERS
from shared_eval_backend import run_logged_asks

# ================================================================================

# ================================================================================

def save_debug_log(model_name: str, model_path: str, duration: float, prompt_list: List[List[Dict]], responses: List[str]):
    try:
        
        eval_root_dir = os.path.dirname(_current_dir)
        debug_dir = os.path.join(eval_root_dir, "GeneratedKGContent", "_TemplateDebug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        
        
        date_str = time.strftime("%Y%m%d")
        
        safe_model_name = model_name.replace('/', '_')
        filename = f"debug_{safe_model_name}_{date_str}.jsonl"
        
        filepath = os.path.join(debug_dir, filename)
        
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        with open(filepath, 'a', encoding='utf-8') as f:
            for i, (p, r) in enumerate(zip(prompt_list, responses)):
                record = {
                    "timestamp": timestamp,
                    "model": model_name,
                    "模型地址": model_path,
                    "运行时间": f"{duration:.2f}s",
                    "index": i,
                    "input_prompt": p,
                    "output_result": r
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
        print(f"   🐛 [Debug] 输入/输出已追加保存至: {filepath}")
        
    except Exception as e:
        print(f"   ⚠️ [Debug] 保存调试信息失败: {e}")



# ================================================================================

# ================================================================================

class ToolsAskMethod(LLMSinglePromptEndToEnd):
    
    def __init__(
        self,
        model: str = 'gpt-5-nano',
        token: int = 17 * 1024,
        temp: float = 0.3,
        think: int = 2,
        check_cache: bool = True,
        flex: bool = False,
        runtime_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            name=f"ToolsAsk({model})",
            token=token,
            temp=temp,
            check_cache=check_cache,
            runtime_context=runtime_context,
            **kwargs,
        )
        self.model = model
        self.think = think
        self.flex = flex
        
        
        try:
            import tools
            self.tools = tools
            print(f"✅ ToolsAskMethod 初始化成功: model={model}, max_tokens={token}")
        except ImportError as e:
            raise ImportError(f"❌ 无法导入 tools 模块: {e}")
    
    def _call_llm(self, prompt_list: List[List[Dict]]) -> List[str]:
        start_time = time.time()
        responses = run_logged_asks(
            prompt_list,
            model=self.model,
            token=self.token,
            temp=self.temp,
            think=self.think,
            runtime_context=self.runtime_context,
            phase="generate",
            prompt_metadata_list=getattr(self, "_grid_prompt_metadata", None),
            check_history_cache=self.check_cache,
            retry=False,
            force_api_do_huge_input_Cloud=True,
            flex=self.flex,
        )
        duration = time.time() - start_time
        
        
        save_debug_log(self.model, "Cloud API", duration, prompt_list, responses)
        
        return [r if r else '' for r in responses]


# ================================================================================

# ================================================================================

class VLLMServerMethod(LLMSinglePromptEndToEnd):
    
    def __init__(
        self,
        model: str = 'local',
        vllm_servers: List[str] = None,
        token: int = 17 * 1024,
        temp: float = 0.2,   
        think: int = 0,
        check_cache: bool = True,
        model_path: str = None,
        auto_cleanup: bool = False,
        skip_verification: bool = False,
        top_p: float = 1.0,   
        top_k: int = -1,      
        shared_llm_backend: Optional[Dict[str, Any]] = None,
        runtime_context: Optional[Dict[str, Any]] = None,
        stream_stall_seconds: Optional[float] = None,
        request_max_total_seconds: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            name=f"VLLMServer({model})",
            token=token,
            temp=temp,
            check_cache=check_cache,
            runtime_context=runtime_context,
            **kwargs
        )
        self.shared_llm_backend = dict(shared_llm_backend or {})
        self._shared_backend_enabled = bool(self.shared_llm_backend.get("enabled"))
        self.model = self.shared_llm_backend.get("model", model)
        
        
        
        self.vllm_servers = (
            self.shared_llm_backend.get("servers", ["super", "normal", "ultra"])
            if self._shared_backend_enabled
            else (vllm_servers or {'super': 2, 'ultra': 4, 'normal': 1})
        )
        self.ask_max_workers_vllm = self.shared_llm_backend.get("max_workers_vllm", 64)
        self.vllm_server_name = self.shared_llm_backend.get("vllm_server_name")
        self.ask_check_history_cache = self.shared_llm_backend.get("check_history_cache", self.check_cache)
        self.ask_vllm_smart_mode = self.shared_llm_backend.get("smart_mode", True)
        self.ask_prompt_send_weight_vllm = self.shared_llm_backend.get(
            "prompt_send_weight_vllm",
            {"super": 2, "ultra": 4, "normal": 1},
        )
        
        
        
        self.ask_stream_stall_seconds = self.shared_llm_backend.get(
            "stream_stall_seconds",
            stream_stall_seconds,
        )
        self.ask_request_max_total_seconds = self.shared_llm_backend.get(
            "request_max_total_seconds",
            request_max_total_seconds,
        )
        self.think = think
        self.model_path = self.shared_llm_backend.get("model_path", model_path)
        self.auto_cleanup = auto_cleanup
        self.skip_verification = skip_verification
        self._deployed = False
        
        
        self.top_p = str(top_p)
        self.top_k = str(top_k)
        
        
        self.vllm_manager = None
        if self.model_path and not self._shared_backend_enabled:
            self.vllm_manager = VLLMEnvironmentManager(
                model_path=self.model_path,
                skip_verification=skip_verification,
                auto_cleanup=auto_cleanup,
                vllm_servers=self.vllm_servers
            )
        
        
        try:
            import tools
            self.tools = tools
            print(
                f"✅ VLLMServerMethod 初始化成功: model={model}, "
                f"servers={self.vllm_servers}, max_workers={self.ask_max_workers_vllm}"
            )
        except ImportError as e:
            raise ImportError(f"❌ 无法导入 tools 模块: {e}")
    
    def _check_current_model(self, target_model_path: str) -> bool:
        try:
            servers_to_check = self.vllm_servers
            if isinstance(servers_to_check, dict):
                servers_to_check = list(servers_to_check.keys())
            if not servers_to_check:
                return False
                
            check_server = servers_to_check[0]
            
            import tools
            current_model_name = tools.resolve_model_name('local', vllm_server_name=check_server)
            
            print(f"   🔍 当前运行模型 ({check_server}): {current_model_name}")
            
            target_name = os.path.basename(target_model_path)
            
            if target_name in current_model_name:
                print(f"   ✅ 目标模型 '{target_name}' 已经在运行")
                return True
            else:
                print(f"   ⚠️ 模型不匹配 (目标: {target_name}, 当前: {current_model_name})")
                return False
                
        except Exception as e:
            print(f"   ⚠️ 无法检查当前模型: {e}")
            return False
    
    def _wait_for_service_ready(self, max_wait: int = 90):
        print(f"\n⏳ 等待服务就绪 (最多 {max_wait} 秒)...")
        
        start_time = time.time()
        check_interval = 10
        
        while time.time() - start_time < max_wait:
            try:
                target_server = self.vllm_servers
                if isinstance(target_server, dict):
                    target_server = [list(target_server.keys())[0]]
                elif isinstance(target_server, list):
                    target_server = target_server[:1]
                
                responses = self.tools.ask_group_link(
                    prompt_list=[[{"role": "user", "content": "hi"}]],
                    model='local',
                    token=50,
                    temp=0.1,
                    think=0,
                    streamprint=False,
                    max_workers_Vllm=target_server,
                    retry=False
                )
                if responses and responses[0]:
                    elapsed = time.time() - start_time
                    print(f"   ✅ 服务已就绪! (耗时 {elapsed:.1f}s)")
                    return True
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"   ⏳ 等待中... ({elapsed:.0f}s) - {str(e)[:50]}")
            
            time.sleep(check_interval)
        
        print(f"   ⚠️ 服务可能尚未完全准备好，但将继续执行...")
        return False

    def _deploy_model(self, model_path: str):
        if self._shared_backend_enabled:
            print("   ⏩ 共享三机后端已由主评估器统一准备，跳过单方法部署")
            self._deployed = True
            return

        print(f"\n{'='*60}")
        print(f"🚀 自动部署模型: {os.path.basename(model_path)}")
        print(f"{'='*60}")
        
        
        if self.vllm_manager:
            
            if self._check_current_model(model_path):
                print(f"   ⏩ 跳过部署 (模型已经在运行)")
                if self._wait_for_service_ready(max_wait=90):
                    self._deployed = True
                    print(f"\n{'='*60}")
                    print(f"✅ 模型就绪，可以开始生成!")
                    print(f"{'='*60}\n")
                    return
                else:
                    print(f"   ⚠️ 服务响应超时，将尝试重启服务...")
            
            self.vllm_manager.deploy(force=True)
            self._deployed = True
        else:
            raise RuntimeError("未指定 model_path，无法部署")
        
        print(f"\n{'='*60}")
        print(f"✅ 模型部署完成，可以开始生成!")
        print(f"{'='*60}\n")
    
    def cleanup(self):
        import sys
        if sys is None or not hasattr(sys, 'meta_path') or sys.meta_path is None:
            return
        
        if self.vllm_manager and self.auto_cleanup:
            self.vllm_manager.cleanup()
            self._deployed = False
    
    def __del__(self):
        try:
            import sys
            if sys is None or not hasattr(sys, 'meta_path') or sys.meta_path is None:
                return
            self.cleanup()
        except:
            pass
    
    def _call_llm(self, prompt_list: List[List[Dict]]) -> List[str]:
        start_time = time.time()
        extra_kwargs: Dict[str, Any] = {}
        if self.ask_stream_stall_seconds is not None:
            extra_kwargs["stream_stall_seconds"] = self.ask_stream_stall_seconds
        if self.ask_request_max_total_seconds is not None:
            extra_kwargs["request_max_total_seconds"] = self.ask_request_max_total_seconds
        responses = run_logged_asks(
            prompt_list,
            model=self.model,
            token=self.token,
            temp=self.temp,
            think=self.think,
            runtime_context=self.runtime_context,
            phase="generate",
            prompt_metadata_list=getattr(self, "_grid_prompt_metadata", None),
            check_history_cache=self.ask_check_history_cache,
            VllmSmartMode=self.ask_vllm_smart_mode,
            max_workers_Vllm=self.ask_max_workers_vllm,
            prompt_send_weight_VllmNotSmartMode=self.ask_prompt_send_weight_vllm,
            vllm_server_name=self.vllm_server_name,
            retry=False,
            force_api_do_huge_input_Cloud=True,
            top_p=self.top_p,
            top_k=self.top_k,
            extra_kwargs=extra_kwargs or None,
        )
        duration = time.time() - start_time
        
        
        save_debug_log(self.model, self.model_path if self.model_path else "Existing VLLM", duration, prompt_list, responses)
        
        return [r if r else '' for r in responses]
    
    def _check_all_cache_with_retries(self, contents: List[str], max_retries: int = 3) -> tuple:
        print(f"\n🔍 缓存优先检查（逐级）...")
        
        
        model_name_for_cache = None
        if self.model_path:
            model_name_for_cache = os.path.basename(self.model_path)
            print(f"   🧠 使用模型名: {model_name_for_cache}")
        
        content_valid_cache = {}
        content_invalid_cache = {}
        pending_indices = list(range(len(contents)))
        
        base_prompts = [self._create_prompt(c) for c in contents]
        
        for attempt in range(max_retries):
            if not pending_indices:
                break
            
            prompts_to_check = []
            for idx in pending_indices:
                retry_prompt = self._create_retry_prompt(base_prompts[idx], attempt)
                prompts_to_check.append(retry_prompt)
            
            suffix_desc = (
                f"(attempt={attempt}, 后缀='{RETRY_SUFFIXES[min(attempt, len(RETRY_SUFFIXES)-1)]}')"
                if attempt > 0 else "(基础版本)"
            )
            print(f"   📦 检查 {len(prompts_to_check)} 个请求 {suffix_desc}...")
            
            
            
            cache_batch_ret = self.tools.check_cache_batch(
                prompt_list=prompts_to_check,
                model=self.model,
                token=self.token,
                temp=self.temp,
                think=self.think,
                max_workers_Vllm=self.ask_max_workers_vllm,
                model_name_override=model_name_for_cache,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            if isinstance(cache_batch_ret, tuple) and len(cache_batch_ret) >= 3:
                cached_results, cached_idx, uncached_idx = cache_batch_ret[:3]
            else:
                raise RuntimeError(f"check_cache_batch 返回格式异常: {type(cache_batch_ret)} / {cache_batch_ret!r}")
            
            
            if attempt == 0 and len(cached_results) == 0:
                print(f"      ⚡ 基础版本全部未命中，跳过后缀检查，直接进入 LLM 请求")
                break
            
            still_pending = []
            for i, orig_idx in enumerate(pending_indices):
                if i in cached_results:
                    response = cached_results[i]
                    parsed = self._robust_json_parse(self._clean_response(response) if response else '')
                    if not parsed.get('error'):
                        content_valid_cache[orig_idx] = response
                    else:
                        content_invalid_cache[orig_idx] = response
                        still_pending.append(orig_idx)
                else:
                    still_pending.append(orig_idx)
            
            valid_count = len(content_valid_cache)
            invalid_count = len([k for k in content_invalid_cache if k in pending_indices])
            if invalid_count > 0:
                print(f"      ✅ 可解析: {valid_count}, ⚠️ 不可解析: {invalid_count}, 仍需检查: {len(still_pending)}")
            else:
                print(f"      ✅ 可解析: {valid_count}/{len(contents)}, 仍需检查: {len(still_pending)}")
            
            pending_indices = still_pending
        
        all_handled = len(content_valid_cache) == len(contents)
        
        if all_handled:
             print(f"🎉 所有 {len(contents)} 个请求均有可解析缓存，无需启动 vLLM 服务！")
        else:
             print(f"   ⚠️ {len(contents) - len(content_valid_cache)} 个请求未找到可解析缓存，将触发 LLM 生成")

        return all_handled, content_valid_cache, content_invalid_cache

    def batch_generate(self, contents: List[str], max_retries: int = 3, **kwargs) -> List[Dict[str, Any]]:
        print(f"🚀 批量生成 {len(contents)} 篇文章的知识图谱...")

        
        if self.check_cache:
            all_handled, content_valid_cache, content_invalid_cache = self._check_all_cache_with_retries(
                contents, max_retries
            )
            
            if all_handled:
                results = [None] * len(contents)
                
                for idx, response in content_valid_cache.items():
                    clean_resp = self._clean_response(response)
                    results[idx] = self._robust_json_parse(clean_resp)
                
                success_count = sum(1 for r in results if r and r.get('relations'))
                print(f"✅ 批量生成完成 (全部来自缓存): {success_count}/{len(contents)} 成功")
                
                return results
        
        
        if self.model_path and not self._deployed and not self._shared_backend_enabled:
            self._deploy_model(self.model_path)
        
        
        return super().batch_generate(contents, max_retries=max_retries, **kwargs)


# ================================================================================

# ================================================================================

def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(
        description='🚀 端到端知识图谱生成 (支持云端 API 和本地 vLLM)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group_input = parser.add_argument_group('输入设置')
    group_input.add_argument('--content', '-c', type=str, help='文章内容 (直接传入)')
    group_input.add_argument('--file', '-f', type=str, help='文章内容文件路径')
    group_input.add_argument('--dir', '-d', type=str, help='文章文件夹路径 (处理所有 .txt)')
    group_input.add_argument('--data_json', type=str, help='标准数据集 JSON 路径')
    
    
    group_model = parser.add_argument_group('模型设置')
    group_model.add_argument('--model', '-m', type=str, default='gpt-5-nano', help='模型名称')
    group_model.add_argument('--model_path', '-p', type=str, help='vLLM 模型权重路径 (本地模式)')
    group_model.add_argument('--token', type=int, default=32*1024, help='最大生成 token (默认 32k)')
    group_model.add_argument('--temp', type=float, default=0.3, help='温度参数')
    group_model.add_argument('--think', type=int, default=2, help='思考深度 (0-3)')
    group_model.add_argument('--no-cache', action='store_true', help='禁用缓存')
    group_model.add_argument('--flex', action='store_true', help='启用 Flex 模式')
    group_model.add_argument('--auto-cleanup', action='store_true', help='完成后自动停止 vLLM 服务')
    group_model.add_argument('--skip-verification', action='store_true', help='跳过三机模型一致性验证')
    
    
    group_eval = parser.add_argument_group('评分设置')
    group_eval.add_argument('--eval', action='store_true', help='是否在生成后立即进行评分')
    group_eval.add_argument('--judge_model', type=str, default='gpt-5-nano', help='评估裁判模型')

    args = parser.parse_args()
    
    
    data_items = []
    if args.data_json:
        if os.path.exists(args.data_json):
            with open(args.data_json, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                data_items = raw_data if isinstance(raw_data, list) else [raw_data]
    elif args.dir:
        files = glob.glob(os.path.join(args.dir, "*.txt"))
        for f_path in files:
            with open(f_path, 'r', encoding='utf-8') as f:
                data_items.append({'content': f.read(), 'source': os.path.basename(f_path)})
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            data_items.append({'content': f.read(), 'source': os.path.basename(args.file)})
    elif args.content:
        data_items.append({'content': args.content, 'source': 'manual_input'})
    
    if not data_items:
        print("❌ 未提供输入内容。")
        return
    
    
    if args.model_path:
        
        method = VLLMServerMethod(
            model='local',
            model_path=args.model_path,
            token=args.token,
            temp=args.temp,
            check_cache=not args.no_cache,
            auto_cleanup=args.auto_cleanup,
            skip_verification=args.skip_verification
        )
    else:
        
        method = ToolsAskMethod(
            model=args.model,
            token=args.token,
            temp=args.temp,
            think=args.think,
            check_cache=not args.no_cache,
            flex=args.flex
        )
    
    try:
        contents = [item['content'] for item in data_items]
        predictions = method.batch_generate(contents)
        
        
        if args.eval:
            print(f"\n✨ 开始全量并发评分 (裁判模型: {args.judge_model})...")
            try:
                import pandas as pd
                _parent_dir = os.path.dirname(_current_dir)
                if _parent_dir not in sys.path:
                    sys.path.append(_parent_dir)
                _eval_dir = os.path.join(_parent_dir, "eval")
                if _eval_dir not in sys.path:
                    sys.path.append(_eval_dir)
                from unified_eval_executor import KgRewardEvaluator

                evaluator = KgRewardEvaluator(judge_model=args.judge_model)
                results = evaluator.batch_evaluate(data_items, predictions)
                
                df = pd.DataFrame(results)
                print("\n" + "=" * 60)
                print("📊 评分结果汇总:")
                print(f"平均 F1: {df['f1'].mean():.4f}")
                print("=" * 60)
                
                save_path = "batch_eval_results.json"
                with open(save_path, 'w', encoding='utf-8') as f:
                    output_data = []
                    for item, pred, res in zip(data_items, predictions, results):
                        combined = item.copy()
                        combined.update({'prediction': pred, 'evaluation': res})
                        output_data.append(combined)
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"💾 结果已保存至: {save_path}")
            except Exception as e:
                print(f"❌ 评分出错: {e}")
        else:
            print("\n" + "=" * 60)
            print(f"✅ 已完成 {len(predictions)} 条生成")
            print("=" * 60)
    finally:
        if hasattr(method, 'cleanup'):
            method.cleanup()


if __name__ == "__main__":
    main()
