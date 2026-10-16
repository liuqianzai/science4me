import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Tuple


class SandboxExecutor:
    """子进程隔离代码执行沙箱"""

    @staticmethod
    def run_script(
        code: str,
        work_dir: str,
        script_name: str = "experiment.py",
        timeout_seconds: int = 300
    ) -> Tuple[bool, str, Dict[str, Any], str]:
        """
        在指定工作目录下写入并运行 Python 脚本
        返回: (是否成功, 日志输出, 指标数据字典, 错误栈信息)
        """
        os.makedirs(work_dir, exist_ok=True)
        script_path = os.path.join(work_dir, script_name)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        metrics_file = os.path.join(work_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                os.remove(metrics_file)
            except OSError:
                pass

        start_time = time.time()
        logs = []
        error_tb = ""
        success = False

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.path.abspath(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

        try:
            process = subprocess.run(
                [sys.executable, script_name],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
                encoding="utf-8",
                errors="replace"
            )
            duration = time.time() - start_time
            logs.append(f"=== [EXECUTION FINISHED IN {duration:.2f}s, EXIT CODE: {process.returncode}] ===")
            if process.stdout:
                logs.append("[STDOUT]:\n" + process.stdout)
            if process.stderr:
                logs.append("[STDERR]:\n" + process.stderr)

            if process.returncode == 0:
                success = True
            else:
                error_tb = process.stderr.strip() or f"Process failed with exit code {process.returncode}"

        except subprocess.TimeoutExpired as e:
            duration = time.time() - start_time
            logs.append(f"=== [EXECUTION TIMEOUT AFTER {duration:.2f}s] ===")
            error_tb = f"Execution exceeded timeout limit ({timeout_seconds} seconds)."
            if e.stdout:
                logs.append("[STDOUT BEFORE TIMEOUT]:\n" + (e.stdout if isinstance(e.stdout, str) else e.stdout.decode("utf-8", "ignore")))
            if e.stderr:
                logs.append("[STDERR BEFORE TIMEOUT]:\n" + (e.stderr if isinstance(e.stderr, str) else e.stderr.decode("utf-8", "ignore")))
        except Exception as e:
            error_tb = f"Subprocess launch error: {str(e)}"
            logs.append(f"[EXCEPTION]: {error_tb}")

        # 读取生成的指标文件
        metrics = {}
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as mf:
                    metrics = json.load(mf)
            except Exception as ex:
                logs.append(f"[WARNING]: Failed to parse metrics.json: {str(ex)}")

        full_log = "\n".join(logs)
        return success, full_log, metrics, error_tb

    @staticmethod
    def run_script_stream(
        code: str,
        work_dir: str,
        script_name: str = "experiment.py",
        timeout_seconds: int = 300,
        log_callback=None
    ) -> Tuple[bool, str, Dict[str, Any], str]:
        """
        在指定工作目录下运行 Python 脚本，支持实时逐行回调推送日志
        返回: (是否成功, 完整日志输出, 指标数据字典, 错误栈信息)
        """
        os.makedirs(work_dir, exist_ok=True)
        script_path = os.path.join(work_dir, script_name)

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        metrics_file = os.path.join(work_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                os.remove(metrics_file)
            except OSError:
                pass

        start_time = time.time()
        logs = []
        error_tb = ""
        success = False

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.path.abspath(work_dir) + os.pathsep + env.get("PYTHONPATH", "")

        try:
            process = subprocess.Popen(
                [sys.executable, script_name],
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                encoding="utf-8",
                errors="replace",
                bufsize=1
            )

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    stripped = line.rstrip("\r\n")
                    logs.append(stripped)
                    if log_callback:
                        try:
                            log_callback(stripped)
                        except Exception:
                            pass

                if time.time() - start_time > timeout_seconds:
                    process.kill()
                    timeout_msg = f"[EXECUTION TIMEOUT AFTER {timeout_seconds}s]"
                    logs.append(timeout_msg)
                    if log_callback:
                        log_callback(timeout_msg)
                    error_tb = f"Execution exceeded timeout limit ({timeout_seconds} seconds)."
                    break

            returncode = process.poll() or 0
            duration = time.time() - start_time
            finish_msg = f"=== [EXECUTION FINISHED IN {duration:.2f}s, EXIT CODE: {returncode}] ==="
            logs.append(finish_msg)
            if log_callback:
                log_callback(finish_msg)

            if returncode == 0:
                success = True
            elif not error_tb:
                error_tb = f"Process failed with exit code {returncode}"

        except Exception as e:
            error_tb = f"Subprocess launch error: {str(e)}"
            logs.append(f"[EXCEPTION]: {error_tb}")
            if log_callback:
                log_callback(f"[EXCEPTION]: {error_tb}")

        # 读取生成的指标文件
        metrics = {}
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r", encoding="utf-8") as mf:
                    metrics = json.load(mf)
            except Exception as ex:
                logs.append(f"[WARNING]: Failed to parse metrics.json: {str(ex)}")

        full_log = "\n".join(logs)
        return success, full_log, metrics, error_tb

