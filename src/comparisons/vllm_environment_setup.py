# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import importlib.util
from typing import Dict, Optional



if os.environ.get("GRID_FORCE_DISABLE_CUDA_VISIBLE_DEVICES") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


DROPBOX_PATH = os.path.join(os.path.expanduser("~"), 'Dropbox')
if DROPBOX_PATH not in sys.path:
    sys.path.insert(0, DROPBOX_PATH)


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


SCRIPT_DIR = os.path.join(DROPBOX_PATH, "各种配置文件.脚本.秘钥/VLLM和VERL等LLM服务指令")
STOP_SCRIPT = os.path.join(SCRIPT_DIR, "stop.sh")
LLM_DOCKER_SCRIPT = os.path.join(SCRIPT_DIR, "llm-docker.sh")


SERVERS = {
    'super': {'ssh': 'asu-super', 'hdd_path': '/mnt/disk1/liangyi'},
    'normal': {'ssh': 'asu-normal', 'hdd_path': '/media/HDD1/liangyi'},
    'ultra': {'ssh': 'asu-ultra', 'hdd_path': '/data/liangyi'},
}


# ================================================================================

# ================================================================================

class VLLMEnvironmentManager:
    
    
    _instance = None
    _current_model_path = None
    
    def __new__(cls, model_path: str = None, **kwargs):
        if cls._instance is None or cls._current_model_path != model_path:
            cls._instance = super().__new__(cls)
            cls._current_model_path = model_path
        return cls._instance
    
    def __init__(
        self,
        model_path: str,
        skip_verification: bool = False,
        auto_cleanup: bool = False,
        vllm_servers: Dict = None,
    ):
        
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        import threading
        
        self.model_path = model_path
        self.skip_verification = skip_verification
        self.auto_cleanup = auto_cleanup
        
        
        # super:ultra:normal = 2:4:1
        self.vllm_servers = vllm_servers or {'super': 2, 'ultra': 4, 'normal': 1}
        self._deployed = False
        self._initialized = True
        self._lock = threading.Lock()  
        
        print(f"  🖥️ VLLMEnvironmentManager 初始化: {os.path.basename(model_path)}")

    def _target_server_names(self):
        if isinstance(self.vllm_servers, dict):
            servers = list(self.vllm_servers.keys())
        elif isinstance(self.vllm_servers, (list, tuple, set)):
            servers = [str(server) for server in self.vllm_servers]
        else:
            servers = ['super', 'ultra', 'normal']

        normalized = []
        for server in servers:
            if server in SERVERS and server not in normalized:
                normalized.append(server)
        return normalized or ['super', 'ultra', 'normal']
    
    def _run_ssh_command(self, server: str, command: str) -> tuple:
        ssh_host = SERVERS[server]['ssh']
        try:
            result = subprocess.run(
                ['ssh', ssh_host, command],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return '', 'SSH command timeout', 1
        except Exception as e:
            return '', str(e), 1
    
    def verify_model_on_servers(self) -> bool:
        if self.skip_verification:
            print("  ⏩ 跳过目标服务器一致性验证")
            return True

        target_servers = self._target_server_names()
        print(f"\n🔍 验证模型在目标服务器上的一致性...")
        print(f"   模型路径: {self.model_path}")
        
        
        server_paths = {}
        for server, config in SERVERS.items():
            path = self.model_path
            if path.startswith('/mnt/disk1/liangyi'):
                server_paths[server] = path.replace('/mnt/disk1/liangyi', config['hdd_path'])
            elif path.startswith('/media/HDD1/liangyi'):
                server_paths[server] = path.replace('/media/HDD1/liangyi', config['hdd_path'])
            elif path.startswith('/data/liangyi'):
                server_paths[server] = path.replace('/data/liangyi', config['hdd_path'])
            else:
                server_paths[server] = path
        
        
        md5_results = {}
        for server in target_servers:
            path = server_paths[server]
            config_path = os.path.join(path, 'config.json')
            
            stdout, stderr, code = self._run_ssh_command(
                server, 
                f"[ -f '{config_path}' ] && md5sum '{config_path}' | cut -d' ' -f1 || echo 'NOT_FOUND'"
            )
            
            if code != 0 or stdout == 'NOT_FOUND':
                print(f"   ❌ {server}: 模型不存在或无法访问 ({config_path})")
                return False
            
            md5_results[server] = stdout
            print(f"   ✅ {server}: {stdout[:8]}... ({path})")
        
        unique_md5 = set(md5_results.values())
        if len(unique_md5) == 1:
            print(f"   ✅ 目标服务器模型一致 (MD5: {list(unique_md5)[0][:16]}...)")
            return True
        else:
            print(f"   ❌ 模型不一致! MD5 不匹配")
            return False
    
    def stop_gpu_services(self):
        target_servers = self._target_server_names()
        print(f"\n🛑 正在终止目标服务器上的 GPU 进程 (>50% VRAM)...")
        try:
            args = ['bash', STOP_SCRIPT]
            for server in target_servers:
                args.append(f'-{server}')
            args.extend(['-percent', '50'])
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                print("   ✅ GPU 进程已终止")
            else:
                print(f"   ⚠️ stop.sh 返回非零: {result.stderr}")
        except Exception as e:
            print(f"   ❌ 执行 stop.sh 失败: {e}")
            raise
    
    def start_vllm_service(self):
        print(f"\n🚀 正在启动 vLLM 服务...")
        print(f"   模型: {self.model_path}")
        
        try:
            target_servers = self._target_server_names()
            args = ['bash', LLM_DOCKER_SCRIPT, '--model-path', self.model_path]
            for server in target_servers:
                args.append(f'-{server}')
            args.extend(['-y', '--detach'])
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("   ✅ vLLM 服务启动成功")
            else:
                print(f"   ❌ vLLM 服务启动失败")
                raise RuntimeError("vLLM service failed to start")
                
        except subprocess.TimeoutExpired:
            print("   ❌ vLLM 服务启动超时 (5分钟)")
            raise
        except Exception as e:
            print(f"   ❌ 执行 llm-docker.sh 失败: {e}")
            raise
    
    def wait_for_ready(self, max_wait: int = 120) -> bool:
        target_servers = self._target_server_names()
        print(f"\n⏳ 等待服务就绪 (最多 {max_wait} 秒, 目标: {target_servers})...")

        start_time = time.time()
        check_interval = 10

        while time.time() - start_time < max_wait:
            ready_servers = []
            status_lines = []

            for server in target_servers:
                model_name = None
                metrics_ok = False
                metrics_error = ""

                try:
                    model_name = tools.llmname(server, shortname=True)
                except Exception as exc:
                    metrics_error = f"llmname={type(exc).__name__}"

                try:
                    stats = tools.get_vllm_realtime_stats(server)
                    metrics_ok = bool(stats.get("success"))
                    if not metrics_ok and not metrics_error:
                        metrics_error = str(stats.get("error", ""))[:80]
                except Exception as exc:
                    if not metrics_error:
                        metrics_error = f"metrics={type(exc).__name__}"

                if model_name:
                    ready_servers.append(server)
                    waiting = None
                    try:
                        waiting = stats.get("waiting") if isinstance(stats, dict) else None
                    except Exception:
                        waiting = None
                    if waiting is not None:
                        status_lines.append(f"{server}=READY({model_name}, waiting={waiting})")
                    else:
                        status_lines.append(f"{server}=READY({model_name})")
                else:
                    state = "models=down"
                    if metrics_ok:
                        state = "metrics=up,models=down"
                    elif metrics_error:
                        state = metrics_error
                    status_lines.append(f"{server}=WAIT({state})")

            elapsed = time.time() - start_time
            print(f"   ⏳ {elapsed:.0f}s -> " + ", ".join(status_lines))

            if len(ready_servers) == len(target_servers):
                print(f"   ✅ 服务已就绪! (耗时 {elapsed:.1f}s)")
                return True

            time.sleep(check_interval)

        print(f"   ⚠️ 服务未在 {max_wait} 秒内全部就绪")
        return False
    
    def is_model_running_on_servers(self) -> bool:
        expected_model_name = os.path.basename(self.model_path.rstrip('/'))
        print(f"   🔍 检查当前运行模型 (期望: {expected_model_name})...")
        
        
        servers = self._target_server_names()
        
        all_match = True
        for server in servers:
            try:
                
                model_name = tools.llmname(server, shortname=True)
                if model_name:
                    model_basename = os.path.basename(model_name.rstrip('/'))
                    if model_basename == expected_model_name or model_name == expected_model_name:
                        print(f"      ✅ {server}: {model_basename}")
                        continue
                    else:
                        print(f"      ❌ {server}: 运行的是 '{model_basename}'，不是期望的模型")
                        all_match = False
                else:
                    print(f"      ❌ {server}: 无模型加载或无法连接")
                    all_match = False
            except Exception as e:
                print(f"      ❌ {server}: 无法连接 ({type(e).__name__})")
                all_match = False
        
        if all_match:
            print(f"   ✅ 所有服务器均已加载目标模型，跳过重启流程")
            return True
        else:
            print(f"   ℹ️ 部分或全部服务器未加载目标模型，需要执行重启")
            return False

    def deploy(self, force: bool = False):
        
        with self._lock:
            if self._deployed and not force:
                print(f"  ⏩ vLLM 已部署，跳过")
                return
            
            print(f"\n{'='*60}")
            print(f"🚀 部署 vLLM 环境: {os.path.basename(self.model_path)}")
            print(f"{'='*60}")
            
            if not self.verify_model_on_servers():
                raise RuntimeError("模型在目标服务器上不存在或不一致")
            
            
            if not force and self.is_model_running_on_servers():
                print("   ✅ 目标模型已经在运行，跳过重启流程")
                print("   ⏩ 跳过停止和启动步骤")
                
                
                
                
                current_wait = 240
                if self.wait_for_ready(max_wait=current_wait):
                    self._deployed = True
                    print(f"\n{'='*60}")
                    print(f"✅ vLLM 环境就绪 (复用现有服务)! 可以使用 model='local' 调用")
                    print(f"{'='*60}\n")
                    return
                else:
                    print("   ⚠️ 复用服务检查失败，准备重启服务...")
            
            self.stop_gpu_services()
            self.start_vllm_service()
            if not self.wait_for_ready(max_wait=240):
                raise RuntimeError(
                    f"vLLM 服务未在预期时间内就绪: {self._target_server_names()} / {self.model_path}"
                )
            
            self._deployed = True
            print(f"\n{'='*60}")
            print(f"✅ vLLM 环境准备完成! 可以使用 model='local' 调用")
            print(f"{'='*60}\n")
    
    def ensure_ready(self):
        if not self._deployed:
            self.deploy()
    
    def cleanup(self):
        if self.auto_cleanup and self._deployed:
            print("\n🧹 正在清理资源，停止 vLLM 服务...")
            self.stop_gpu_services()
            self._deployed = False
            print("   ✅ 清理完成")
    
    @property
    def is_deployed(self) -> bool:
        return self._deployed


# ================================================================================

# ================================================================================

def prepare_vllm_environment(model_path: str, skip_verification: bool = False) -> VLLMEnvironmentManager:
    manager = VLLMEnvironmentManager(model_path, skip_verification=skip_verification)
    manager.deploy()
    return manager


# ================================================================================

# ================================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='🔧 vLLM 环境管理工具 - 一键部署 vLLM Docker 服务',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 部署模型
  python vllm_environment_setup.py --model-path /mnt/disk1/liangyi/models/Qwen3-4B
  
  # 跳过验证快速部署
  python vllm_environment_setup.py --model-path /path/to/model --skip-verification
  
  # 仅停止现有服务
  python vllm_environment_setup.py --stop-only
        """
    )
    
    parser.add_argument('--model-path', '-p', type=str, help='模型权重路径')
    parser.add_argument('--skip-verification', action='store_true', help='跳过三机一致性验证')
    parser.add_argument('--stop-only', action='store_true', help='仅停止现有 GPU 服务')
    
    args = parser.parse_args()
    
    if args.stop_only:
        print("🛑 停止所有 GPU 服务...")
        manager = VLLMEnvironmentManager.__new__(VLLMEnvironmentManager)
        manager.stop_gpu_services()
        print("✅ 完成")
        return
    
    if not args.model_path:
        parser.print_help()
        print("\n❌ 请指定 --model-path 参数")
        return
    
    prepare_vllm_environment(args.model_path, skip_verification=args.skip_verification)


if __name__ == "__main__":
    main()
