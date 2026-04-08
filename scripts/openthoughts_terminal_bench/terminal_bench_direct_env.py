import base64
import hashlib
import inspect
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from rllm.environments.base.base_env import BaseEnv
from rllm.tools.tool_base import Tool


TERMINAL_BENCH_OPENHANDS_SYSTEM_PROMPT = """You are an OpenHands-style terminal agent working inside a Linux task container.

Solve the user's task by taking concrete actions in the terminal instead of only describing what should be done.

Rules:
- Your working directory is /app inside the task container.
- Use execute_bash for shell commands, installs, tests, and quick inspection.
- Use str_replace_editor for viewing, creating, and editing files.
- Use one tool call at a time and wait for the result before making the next tool call.
- Prefer short, verifiable commands and inspect the results before moving on.
- For code tasks, inspect the relevant source files and tests, edit files in /app, and then verify.
- Task tests are mounted under /tests, not /app/tests.
- Use think when you need to reason without changing the environment.
- Do not paste large code blocks in plain text. Use str_replace_editor or execute_bash to write files.
- If a command fails, adapt and try a different approach.
- Do not stop at analysis. Make the necessary filesystem changes in the container.
- When you believe the task is solved, call finish. That will run the task verifier against the current container state.

Be concise, action-oriented, and persistent.
"""


class ExecuteBashTool(Tool):
    def __init__(self, name: str | None = None, description: str | None = None):
        super().__init__(
            name=name or "execute_bash",
            description=description or "Execute a bash command inside the task container at /app and return stdout, stderr, and exit code.",
        )

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Execute a bash command in the task container. Use it for shell inspection, installs, "
                    "running tests, and command-line file operations. You can only execute one bash command "
                    "at a time; if you need multiple sequential actions, chain them with && or ;."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": (
                                "The bash command to run inside the container. You can only execute one "
                                "bash command at a time. If you need multiple sequential actions, chain "
                                "them with && or ;."
                            ),
                        },
                        "is_input": {
                            "type": "string",
                            "enum": ["true", "false"],
                            "description": "Optional OpenHands-compatible flag for sending input to a running process. This env currently treats commands as regular bash commands.",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Optional timeout in seconds for this command.",
                        },
                        "security_risk": {
                            "type": "string",
                            "description": "Optional OpenHands-compatible risk label.",
                        },
                    },
                    "required": ["command"],
                },
            },
        }


class StrReplaceEditorTool(Tool):
    def __init__(self, name: str | None = None, description: str | None = None):
        super().__init__(
            name=name or "str_replace_editor",
            description=description
            or "View, create, and edit files using OpenHands-style string replacement operations.",
        )

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Custom editing tool for viewing, creating, and editing files in plain-text format. "
                    "Use it to inspect files, create new files, replace exact text, insert text by line, "
                    "or undo the last edit."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                            "description": "The editor command to run.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Absolute path to a file or directory inside the task container.",
                        },
                        "file_text": {
                            "type": "string",
                            "description": "Required for create. Full content of the file to create.",
                        },
                        "old_str": {
                            "type": "string",
                            "description": "Required for str_replace. Exact text to replace.",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "Replacement text for str_replace or inserted text for insert.",
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": "Required for insert. Insert new_str after this 1-based line number.",
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Optional for view on files. [start, end] inclusive, use -1 for end of file.",
                        },
                        "security_risk": {
                            "type": "string",
                            "description": "Optional OpenHands-compatible risk label.",
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        }


class ThinkTool(Tool):
    def __init__(self, name: str | None = None, description: str | None = None):
        super().__init__(
            name=name or "think",
            description=description or "Log reasoning without changing the environment.",
        )

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Use this tool to reason about the task without changing files or running commands."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "The reasoning to log.",
                        }
                    },
                    "required": ["thought"],
                },
            },
        }


class FinishTool(Tool):
    def __init__(self, name: str | None = None, description: str | None = None):
        super().__init__(
            name=name or "finish",
            description=description or "Finish the task and run verification.",
        )

    @property
    def json(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Signal task completion and run the verifier against the current container state.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Short summary of what you changed and why the task should now pass.",
                        }
                    },
                    "required": ["message"],
                },
            },
        }


TERMINAL_BENCH_TOOL_MAP = {
    "execute_bash": ExecuteBashTool,
    "str_replace_editor": StrReplaceEditorTool,
    "think": ThinkTool,
    "finish": FinishTool,
}


def _truncate_display_text(text: str, max_lines: int = 80, max_chars: int = 2000) -> str:
    if not text:
        return text

    lines = text.splitlines()
    truncated = False
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    clipped = "\n".join(lines)
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars].rstrip()
        truncated = True

    if truncated:
        clipped += "\n[output truncated]"
    return clipped


class TerminalBenchDirectEnv(BaseEnv):
    _kube_namespace_cache: set[str] = set()
    _kube_namespace_lock = threading.Lock()
    _kube_control_parallelism = max(1, int(os.environ.get("TB_KUBE_CONTROL_MAX_PARALLEL", "48")))
    _kube_control_gate = threading.BoundedSemaphore(_kube_control_parallelism)

    def __init__(
        self,
        task_name: str | None = None,
        task_root: str | None = None,
        harbor_host: str | None = None,
        harbor_port: int | None = None,
        harbor_user: str | None = None,
        harbor_ssh_key: str | None = None,
        tasks_root: str | None = None,
        step_timeout: int | None = None,
        verifier_timeout: int | None = None,
        keep_container: bool | None = None,
        container_name_prefix: str | None = None,
        container_env: dict[str, str] | None = None,
        debug_log_path: str | None = None,
        max_steps: int | None = None,
        execution_backend: str | None = None,
        kubeconfig_path: str | None = None,
        kube_namespace: str | None = None,
        kube_node_name: str | None = None,
        kube_ready_timeout: int | None = None,
        **task: Any,
    ):
        self.task = dict(task)
        self.task_name = task_name or self.task.get("task_name")
        self.tasks_root = tasks_root or self.task.get("tasks_root") or os.environ.get("TB_TASKS_ROOT", "/data/terminal-bench-2")
        self.task_root = task_root or self.task.get("task_root") or (
            os.path.join(self.tasks_root, self.task_name) if self.task_name else None
        )

        self.harbor_host = harbor_host or os.environ.get("HARBOR_HOST", "118.196.87.89")
        self.harbor_port = int(harbor_port or self.task.get("harbor_port") or os.environ.get("HARBOR_PORT", "7622"))
        self.harbor_user = harbor_user or os.environ.get("HARBOR_USER", "root")
        self.harbor_ssh_key = harbor_ssh_key or os.environ.get("HARBOR_SSH_KEY", "/root/.ssh/k8s-node.pem")
        self.step_timeout = int(step_timeout or self.task.get("step_timeout") or os.environ.get("TB_STEP_TIMEOUT", "120"))
        self.verifier_timeout = int(
            verifier_timeout or self.task.get("verifier_timeout") or os.environ.get("TB_VERIFIER_TIMEOUT", "1800")
        )
        self.keep_container = _to_bool(keep_container if keep_container is not None else self.task.get("keep_container"), False)
        self.container_name_prefix = container_name_prefix or self.task.get("container_name_prefix") or "rllm-tb-direct"
        self.container_env = dict(container_env or self.task.get("container_env") or _default_container_env())
        self.debug_log_path = debug_log_path or self.task.get("debug_log_path") or os.environ.get("TB_DIRECT_DEBUG_LOG")
        self.max_steps = int(max_steps or self.task.get("max_steps") or os.environ.get("TB_MAX_STEPS", "8"))
        self.execution_backend = (
            execution_backend or self.task.get("execution_backend") or os.environ.get("TB_EXECUTION_BACKEND", "ssh")
        ).strip().lower()
        self.kubeconfig_path = (
            kubeconfig_path or self.task.get("kubeconfig_path") or os.environ.get("TB_KUBECONFIG", "/data/k8s_access/kubeconfig")
        )
        self.kube_namespace = (
            kube_namespace or self.task.get("kube_namespace") or os.environ.get("TB_KUBE_NAMESPACE", "terminal-bench")
        )
        self.kube_ready_timeout = int(
            kube_ready_timeout
            or self.task.get("kube_ready_timeout")
            or os.environ.get("TB_KUBE_READY_TIMEOUT", "600")
        )
        self.kube_image_pull_policy = os.environ.get("TB_KUBE_IMAGE_PULL_POLICY", "IfNotPresent").strip() or "IfNotPresent"
        self.kube_image_registry = os.environ.get("TB_KUBE_IMAGE_REGISTRY", "").strip()
        self.kube_image_map = os.environ.get("TB_KUBE_IMAGE_MAP", "").strip()
        self.kube_node_name = kube_node_name if kube_node_name is not None else self.task.get("kube_node_name")
        if self.kube_node_name is None:
            self.kube_node_name = os.environ.get("TB_KUBE_NODE_NAME")
        if isinstance(self.kube_node_name, str):
            self.kube_node_name = self.kube_node_name.strip() or None
        self._ssh_control_path = self._build_ssh_control_path()

        self.container_name: str | None = None
        self.log_dir: str | None = None
        self.step_count = 0
        self.task_config: dict[str, Any] | None = None
        self.edit_history: dict[str, list[str]] = {}

    def reset(self, task: dict | None = None) -> tuple[dict[str, str], dict[str, Any]]:
        self._log("reset:start", task_name=self.task_name, task_root=self.task_root)
        if task is not None:
            merged = dict(task)
            merged.update(
                {
                    "harbor_host": self.harbor_host,
                    "harbor_port": self.harbor_port,
                    "harbor_user": self.harbor_user,
                    "harbor_ssh_key": self.harbor_ssh_key,
                    "tasks_root": self.tasks_root,
                    "step_timeout": self.step_timeout,
                    "verifier_timeout": self.verifier_timeout,
                    "keep_container": self.keep_container,
                    "container_name_prefix": self.container_name_prefix,
                    "debug_log_path": self.debug_log_path,
                    "max_steps": self.max_steps,
                    "execution_backend": self.execution_backend,
                    "kubeconfig_path": self.kubeconfig_path,
                    "kube_namespace": self.kube_namespace,
                    "kube_node_name": self.kube_node_name,
                    "kube_ready_timeout": self.kube_ready_timeout,
                }
            )
            refreshed = self.from_dict(merged)
            self.__dict__.update(refreshed.__dict__)

        if not self.task_name:
            raise RuntimeError("task_name is required for TerminalBenchDirectEnv")
        if not self.task_root:
            raise RuntimeError("task_root is required for TerminalBenchDirectEnv")

        self.close()
        self.step_count = 0
        self.edit_history = {}
        self.task_config = self._load_task_config()
        k8s_services_config = self._load_k8s_services_config()
        k8s_services = k8s_services_config.get("services", [])
        k8s_task_env = k8s_services_config.get("task_env", {})

        instruction = self._read_remote_file(self._task_file("instruction.md")).strip()
        env_config = self.task_config.get("environment", {})
        verifier_config = self.task_config.get("verifier", {})

        # K8s pod names must be RFC 1123 compliant: lowercase alphanumeric, '-' or '.'
        safe_task_name = self.task_name[:24].replace("_", "-").lower()
        self.container_name = f"{self.container_name_prefix}-{safe_task_name}-{uuid.uuid4().hex[:8]}"
        self.log_dir = f"/tmp/{self.container_name}-logs"

        docker_image = env_config.get("docker_image") or os.environ.get("TB_DEFAULT_DOCKER_IMAGE", "").strip()
        if not docker_image:
            raise RuntimeError(f"{self.task_name} is missing environment.docker_image in task.toml (and TB_DEFAULT_DOCKER_IMAGE is not set)")
        docker_image = self._resolve_kube_image(str(docker_image)) if self.execution_backend == "k8s" else str(docker_image)

        if self.execution_backend == "k8s":
            self._ensure_kube_namespace()
            manifest = self._build_k8s_pod_manifest(
                docker_image=docker_image,
                env_config=env_config,
                k8s_services=k8s_services,
                task_env=k8s_task_env,
            )
            self._run_kubectl(["delete", "pod", self.container_name, "--ignore-not-found=true"], check=False, timeout=60)
            self._run_kubectl(["apply", "-f", "-"], input_text=json.dumps(manifest), timeout=max(self.step_timeout, 60))
            self._run_kubectl(
                ["wait", "--for=condition=Ready", f"pod/{self.container_name}", f"--timeout={self.kube_ready_timeout}s"],
                timeout=max(self.kube_ready_timeout + 30, self.step_timeout, 210),
            )
            self._sync_k8s_task_tests()
            self._sync_k8s_task_seeds()
            self._ensure_k8s_task_dirs()
        else:
            run_parts = [
                "docker run -d",
                f"--name {shlex.quote(self.container_name)}",
                "--workdir /app",
                f"--cpus {shlex.quote(str(env_config.get('cpus', 1)))}",
            ]
            memory = env_config.get("memory")
            if memory:
                run_parts.append(f"--memory {shlex.quote(str(memory))}")
            for env_name, env_value in self.container_env.items():
                if env_value:
                    run_parts.append(f"-e {shlex.quote(f'{env_name}={env_value}')}")
            run_parts.extend(
                [
                    f"-v {shlex.quote(self._task_file('tests'))}:/tests:ro",
                    f"-v {shlex.quote(self.log_dir)}:/logs/verifier",
                    shlex.quote(str(docker_image)),
                    "bash -lc",
                    shlex.quote("trap 'exit 0' TERM INT; while true; do sleep 3600; done"),
                ]
            )

            remote_cmd = "\n".join(
                [
                    f"docker rm -f {shlex.quote(self.container_name)} >/dev/null 2>&1 || true",
                    f"rm -rf {shlex.quote(self.log_dir)}",
                    f"mkdir -p {shlex.quote(self.log_dir)}",
                    " ".join(run_parts),
                ]
            )
            self._run_remote(remote_cmd, timeout=max(self.step_timeout, 60))
        self._log("reset:container_ready", container_name=self.container_name, docker_image=docker_image, log_dir=self.log_dir)

        app_snapshot = self._run_bash("ls -la /app").get("display", "")
        tests_snapshot = self._run_bash("find /tests -maxdepth 2 -type f | sort | head -n 40").get("display", "")

        question = "\n\n".join(
            [
                f"Task: {self.task_name}",
                instruction,
                "You are already inside the task container at /app.",
                "Important: task tests are mounted at /tests, not /app/tests.",
                "Initial /app snapshot:",
                app_snapshot,
                "Initial /tests snapshot:",
                tests_snapshot,
                "Use execute_bash for shell commands and tests.",
                "Use str_replace_editor to view and edit files.",
                "Use think for reasoning only.",
                "When you believe the task is solved, call finish to run the verifier.",
                f"Verifier timeout budget: {verifier_config.get('timeout_sec', self.verifier_timeout)} seconds.",
            ]
        )
        info = {
            "task_name": self.task_name,
            "task_root": self.task_root,
            "container_name": self.container_name,
            "log_dir": self.log_dir,
        }
        return {"question": question}, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.step_count += 1
        self._log("step:start", step_count=self.step_count, action_type=type(action).__name__)

        if isinstance(action, str):
            reward, info = self._run_verifier(summary=action)
            self._log("step:submit_from_string", step_count=self.step_count, reward=reward)
            return {}, reward, True, info

        tool_calls = action or []
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]

        if not isinstance(tool_calls, list):
            raise TypeError(f"Unsupported action type: {type(action).__name__}")

        tool_outputs: dict[str, str] = {}
        info: dict[str, Any] = {"tool_results": []}

        for idx, tool_call in enumerate(tool_calls):
            function = (tool_call or {}).get("function", {})
            tool_name = function.get("name", "")
            raw_args = function.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {"command": raw_args}
            call_id = tool_call.get("id") or f"tool_{idx}"

            try:
                if tool_name == "execute_bash":
                    command = raw_args.get("command") or raw_args.get("cmd")
                    timeout = raw_args.get("timeout")
                    result = self._run_bash(command or "", timeout=timeout)
                    tool_outputs[call_id] = result["display"]
                    info["tool_results"].append(result)
                    self._log("step:execute_bash", step_count=self.step_count, command=command, exit_code=result["exit_code"])
                    continue

                if tool_name == "str_replace_editor":
                    result = self._run_editor(raw_args)
                    tool_outputs[call_id] = result["display"]
                    info["tool_results"].append(result)
                    self._log("step:str_replace_editor", step_count=self.step_count, editor_command=raw_args.get("command"), path=raw_args.get("path"))
                    continue

                if tool_name == "think":
                    thought = (raw_args.get("thought") or "").strip()
                    result = {
                        "tool_name": "think",
                        "thought": thought,
                        "display": "[think]\nThought recorded." if thought else "[think]\n<empty>",
                    }
                    tool_outputs[call_id] = result["display"]
                    info["tool_results"].append(result)
                    self._log("step:think", step_count=self.step_count, thought=_truncate(thought, 300))
                    continue

                if tool_name in {"submit", "finish"}:
                    summary = raw_args.get("summary") or raw_args.get("message") or raw_args.get("response", "")
                    reward, final_info = self._run_verifier(summary=summary)
                    info.update(final_info)
                    self._log("step:submit", step_count=self.step_count, reward=reward, summary=summary)
                    return {}, reward, True, info

                unknown = f"Unknown tool: {tool_name}"
                tool_outputs[call_id] = unknown
                info["tool_results"].append({"tool_name": tool_name, "error": unknown})
            except Exception as exc:
                error_message = str(exc).strip() or exc.__class__.__name__
                display = f"[{tool_name or 'tool'}]\n[stderr]\n{error_message}\n[exit_code] 1"
                result = {
                    "tool_name": tool_name or "tool",
                    "command": raw_args.get("command") if isinstance(raw_args, dict) else None,
                    "path": raw_args.get("path") if isinstance(raw_args, dict) else None,
                    "error": error_message,
                    "display": display,
                    "exit_code": 1,
                }
                tool_outputs[call_id] = display
                info["tool_results"].append(result)
                self._log(
                    "step:tool_error",
                    step_count=self.step_count,
                    tool_name=tool_name,
                    error=error_message,
                    command=result.get("command"),
                    path=result.get("path"),
                )

        done = self.step_count >= self.max_steps
        if done:
            reward, final_info = self._run_verifier(summary="Max steps reached")
            info.update(final_info)
            self._log("step:max_steps_submit", step_count=self.step_count, reward=reward)
            return {}, reward, True, info

        self._log("step:end", step_count=self.step_count, done=False)
        return {"tool_outputs": tool_outputs}, 0.0, False, info

    def close(self):
        self._log("close:start", container_name=self.container_name, log_dir=self.log_dir)
        if self.container_name:
            if self.execution_backend == "k8s":
                self._run_kubectl(["delete", "pod", self.container_name, "--ignore-not-found=true"], check=False, timeout=90)
            else:
                self._run_remote(f"docker rm -f {shlex.quote(self.container_name)} >/dev/null 2>&1 || true", check=False)
        if self.log_dir and not self.keep_container and self.execution_backend != "k8s":
            self._run_remote(f"rm -rf {shlex.quote(self.log_dir)}", check=False)
        self.container_name = None
        self.log_dir = None
        self._log("close:end")

    @staticmethod
    def from_dict(extra_info: dict | str) -> "TerminalBenchDirectEnv":
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        sig = inspect.signature(TerminalBenchDirectEnv.__init__)
        init_params = {}
        consumed_keys = set()
        for param_name in sig.parameters:
            if param_name == "self":
                continue
            if param_name in extra_info:
                init_params[param_name] = extra_info[param_name]
                consumed_keys.add(param_name)
        remaining = {key: value for key, value in extra_info.items() if key not in consumed_keys}
        return TerminalBenchDirectEnv(**init_params, **remaining)

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    def _run_bash(self, command: str, timeout: int | float | None = None) -> dict[str, Any]:
        if not self.container_name:
            raise RuntimeError("Container is not initialized")

        normalized_command = (command or "").strip()
        if not normalized_command:
            payload = {
                "tool_name": "execute_bash",
                "command": command,
                "stdout": "",
                "stderr": "Empty command",
                "exit_code": 2,
                "display": "$ <empty command>\n[stderr]\nEmpty command\n[exit_code] 2",
            }
            self._log("bash:empty")
            return payload

        effective_timeout = int(float(timeout)) if timeout is not None else self.step_timeout
        self._log("bash:start", command=normalized_command, timeout=effective_timeout)
        wrapped_inner = f"cd /app && {normalized_command}"
        if self.execution_backend == "k8s":
            result = self._run_kubectl_exec(
                f"timeout --signal=KILL {shlex.quote(str(effective_timeout))}s bash -lc {shlex.quote(wrapped_inner)}",
                check=False,
                timeout=effective_timeout + 20,
            )
        else:
            exec_cmd = (
                f"timeout --signal=KILL {shlex.quote(str(effective_timeout))}s "
                f"docker exec {shlex.quote(self.container_name)} bash -lc {shlex.quote(wrapped_inner)}"
            )
            result = self._run_remote(exec_cmd, check=False, timeout=effective_timeout + 15)
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        display_stdout = _truncate_display_text(stdout)
        display_stderr = _truncate_display_text(stderr)
        display_parts = [f"$ {normalized_command}"]
        if display_stdout:
            display_parts.append(display_stdout)
        if display_stderr:
            display_parts.append(f"[stderr]\n{display_stderr}")
        display_parts.append(f"[exit_code] {result.returncode}")
        payload = {
            "tool_name": "execute_bash",
            "command": normalized_command,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": result.returncode,
            "display": "\n".join(display_parts).strip(),
        }
        self._log("bash:end", command=normalized_command, exit_code=result.returncode)
        return payload

    def _run_editor(self, args: dict[str, Any]) -> dict[str, Any]:
        command = (args.get("command") or "").strip()
        path = (args.get("path") or "").strip()
        if not command or not path:
            raise RuntimeError("str_replace_editor requires command and path")
        if not path.startswith("/"):
            raise RuntimeError("str_replace_editor path must be absolute")

        if command == "view":
            return self._editor_view(path, args.get("view_range"))
        if command == "create":
            file_text = args.get("file_text")
            if file_text is None:
                raise RuntimeError("create requires file_text")
            return self._editor_create(path, str(file_text))
        if command == "str_replace":
            old_str = args.get("old_str")
            if old_str is None:
                raise RuntimeError("str_replace requires old_str")
            new_str = args.get("new_str", "")
            return self._editor_str_replace(path, str(old_str), str(new_str))
        if command == "insert":
            if "insert_line" not in args:
                raise RuntimeError("insert requires insert_line")
            new_str = args.get("new_str", "")
            return self._editor_insert(path, int(args["insert_line"]), str(new_str))
        if command == "undo_edit":
            return self._editor_undo(path)
        raise RuntimeError(f"Unsupported str_replace_editor command: {command}")

    def _editor_view(self, path: str, view_range: Any = None) -> dict[str, Any]:
        kind = self._container_path_kind(path)
        if kind == "missing":
            display = f"[view {path}]\n[stderr]\nPath does not exist\n[exit_code] 1"
            return {"tool_name": "str_replace_editor", "command": "view", "path": path, "display": display, "exit_code": 1}

        if kind == "dir":
            result = self._run_bash(f"find {shlex.quote(path)} -maxdepth 2 \\( -type f -o -type d \\) | sort")
            display = f"[view {path}]\n{result['display']}"
            result.update({"tool_name": "str_replace_editor", "command": "view", "path": path, "display": display})
            return result

        content = self._read_container_file(path)
        numbered = _with_line_numbers(content)
        if isinstance(view_range, list) and len(view_range) == 2:
            start = max(int(view_range[0]), 1)
            end = int(view_range[1])
            numbered_lines = numbered.splitlines()
            if end == -1:
                selected = numbered_lines[start - 1 :]
            else:
                selected = numbered_lines[start - 1 : end]
            numbered = "\n".join(selected)
        display = f"[view {path}]\n{numbered}".strip()
        return {
            "tool_name": "str_replace_editor",
            "command": "view",
            "path": path,
            "display": display,
            "exit_code": 0,
        }

    def _editor_create(self, path: str, file_text: str) -> dict[str, Any]:
        if self._container_path_kind(path) != "missing":
            raise RuntimeError(f"Cannot create {path}: path already exists")
        self._write_container_file(path, file_text)
        display = f"[create {path}]\nCreated file with {len(file_text.splitlines())} line(s)."
        return {
            "tool_name": "str_replace_editor",
            "command": "create",
            "path": path,
            "display": display,
            "exit_code": 0,
        }

    def _editor_str_replace(self, path: str, old_str: str, new_str: str) -> dict[str, Any]:
        content = self._read_container_file(path)
        matches = content.count(old_str)
        if matches != 1:
            raise RuntimeError(f"str_replace requires exactly one match, found {matches}")
        updated = content.replace(old_str, new_str, 1)
        self._push_edit_history(path, content)
        self._write_container_file(path, updated)
        display = f"[str_replace {path}]\nReplaced 1 occurrence."
        return {
            "tool_name": "str_replace_editor",
            "command": "str_replace",
            "path": path,
            "display": display,
            "exit_code": 0,
        }

    def _editor_insert(self, path: str, insert_line: int, new_str: str) -> dict[str, Any]:
        content = self._read_container_file(path)
        lines = content.splitlines()
        trailing_newline = content.endswith("\n")
        idx = max(insert_line, 0)
        if idx > len(lines):
            raise RuntimeError(f"insert_line {insert_line} is out of range for {path}")
        insert_lines = new_str.splitlines()
        updated_lines = lines[:idx] + insert_lines + lines[idx:]
        updated = "\n".join(updated_lines)
        if trailing_newline or new_str.endswith("\n"):
            updated += "\n"
        self._push_edit_history(path, content)
        self._write_container_file(path, updated)
        display = f"[insert {path}]\nInserted {len(insert_lines)} line(s) after line {insert_line}."
        return {
            "tool_name": "str_replace_editor",
            "command": "insert",
            "path": path,
            "display": display,
            "exit_code": 0,
        }

    def _editor_undo(self, path: str) -> dict[str, Any]:
        history = self.edit_history.get(path) or []
        if not history:
            raise RuntimeError(f"No edit history available for {path}")
        previous = history.pop()
        self._write_container_file(path, previous)
        display = f"[undo_edit {path}]\nRestored previous file contents."
        return {
            "tool_name": "str_replace_editor",
            "command": "undo_edit",
            "path": path,
            "display": display,
            "exit_code": 0,
        }

    def _container_path_kind(self, path: str) -> str:
        command = (
            f"if [ -d {shlex.quote(path)} ]; then echo dir; "
            f"elif [ -f {shlex.quote(path)} ]; then echo file; "
            f"else echo missing; fi"
        )
        result = self._run_bash(command)
        return (result.get("stdout") or "missing").strip()

    def _read_container_file(self, path: str) -> str:
        result = self._run_bash(f"cat {shlex.quote(path)}")
        if result["exit_code"] != 0:
            raise RuntimeError(result["display"])
        return result["stdout"]

    def _write_container_file(self, path: str, content: str) -> None:
        parent = os.path.dirname(path) or "/"
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        command = (
            f"mkdir -p {shlex.quote(parent)} && "
            f"python3 -c {shlex.quote(_python_write_file_script(path, encoded))}"
        )
        result = self._run_bash(command, timeout=max(self.step_timeout, 120))
        if result["exit_code"] != 0:
            raise RuntimeError(result["display"])

    def _push_edit_history(self, path: str, content: str) -> None:
        self.edit_history.setdefault(path, []).append(content)

    def _run_verifier(self, summary: str = "") -> tuple[float, dict[str, Any]]:
        if not self.container_name or not self.log_dir:
            raise RuntimeError("Container is not initialized")

        verifier_timeout = max(
            int(self.task_config.get("verifier", {}).get("timeout_sec", 0)) if self.task_config else 0,
            self.verifier_timeout,
        )
        self._log("verifier:start", summary=summary, timeout=verifier_timeout)
        if self.execution_backend == "k8s":
            verify_inner = (
                "chmod +x /tests/test.sh 2>/dev/null || true; "
                "/bin/bash /tests/test.sh; "
                "status=$?; "
                "if [ -f /logs/verifier/reward.txt ]; then cat /logs/verifier/reward.txt; fi; "
                "exit $status"
            )
            result = self._run_kubectl_exec(
                f"timeout --signal=KILL {shlex.quote(str(verifier_timeout))}s bash -lc {shlex.quote(verify_inner)}",
                check=False,
                timeout=verifier_timeout + 30,
            )
        else:
            verify_cmd = "\n".join(
                [
                    f"timeout --signal=KILL {shlex.quote(str(verifier_timeout))}s docker exec {shlex.quote(self.container_name)} "
                    f"bash -lc {shlex.quote('chmod +x /tests/test.sh 2>/dev/null || true; /bin/bash /tests/test.sh')}",
                    f"if [ -f {shlex.quote(os.path.join(self.log_dir, 'reward.txt'))} ]; then cat {shlex.quote(os.path.join(self.log_dir, 'reward.txt'))}; fi",
                ]
            )
            result = self._run_remote(verify_cmd, check=False, timeout=verifier_timeout + 30)

        reward = 0.0
        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            try:
                reward = float(last)
            except ValueError:
                reward = 1.0 if result.returncode == 0 else 0.0
        elif result.returncode == 0:
            reward = 1.0

        info = {
            "summary": summary,
            "reward": reward,
            "verifier_stdout": (result.stdout or "").strip(),
            "verifier_stderr": (result.stderr or "").strip(),
            "container_name": self.container_name,
            "log_dir": self.log_dir,
        }
        self._log("verifier:end", reward=reward, returncode=result.returncode)
        return reward, info

    def _task_file(self, relative_path: str) -> str:
        if not self.task_root:
            raise RuntimeError("task_root is not set")
        return f"{self.task_root.rstrip('/')}/{relative_path.lstrip('/')}"

    def _load_task_config(self) -> dict[str, Any]:
        raw = self._read_remote_file(self._task_file("task.toml"))
        return tomllib.loads(raw)

    def _load_k8s_services_config(self) -> dict[str, Any]:
        path = self._task_file("k8s_services.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, list):
                    return {"services": raw, "task_env": {}}
                return {"services": raw.get("services", []), "task_env": raw.get("task_env", {})}

        # If the task root is a local directory, don't bother checking remote
        if self.task_root and os.path.isdir(self.task_root):
            return {"services": [], "task_env": {}}

        result = self._run_remote(f"test -f {shlex.quote(path)} && cat {shlex.quote(path)}", check=False, timeout=60)
        if result.returncode != 0 or not (result.stdout or "").strip():
            return {"services": [], "task_env": {}}
        raw = json.loads(result.stdout)
        if isinstance(raw, list):
            return {"services": raw, "task_env": {}}
        return {"services": raw.get("services", []), "task_env": raw.get("task_env", {})}

    def _read_remote_file(self, path: str) -> str:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read()
        result = self._run_remote(f"cat {shlex.quote(path)}", timeout=60)
        return result.stdout

    def _kubectl_base_command(self) -> list[str]:
        kubectl_bin = os.environ.get("TB_KUBECTL_BIN") or "/data/k8s_access/kubectl.real"
        if not os.path.exists(kubectl_bin):
            kubectl_bin = shutil.which("kubectl") or "/usr/local/bin/kubectl"
        return [
            kubectl_bin,
            "--kubeconfig",
            self.kubeconfig_path,
            "-n",
            self.kube_namespace,
        ]

    def _run_kubectl(
        self,
        args: list[str],
        check: bool = True,
        timeout: int = 120,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess:
        attempts = 3
        proc = None
        run_args = list(args)
        for attempt in range(1, attempts + 1):
            self._log("kubectl:start", timeout=timeout, args=run_args, attempt=attempt)
            try:
                with self._kube_control_gate:
                    proc = subprocess.run(
                        [*self._kubectl_base_command(), *run_args],
                        input=input_text,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
            except (BlockingIOError, OSError, RuntimeError) as exc:
                error_text = str(exc)
                self._log("kubectl:spawn_error", error=error_text, attempt=attempt, args=run_args)
                transient_spawn_error = "can't start new thread" in error_text or "Resource temporarily unavailable" in error_text
                if transient_spawn_error and attempt < attempts:
                    time.sleep(min(2 * attempt, 5))
                    continue
                raise
            self._log(
                "kubectl:end",
                returncode=proc.returncode,
                stdout=_truncate(proc.stdout, 300),
                stderr=_truncate(proc.stderr, 300),
                attempt=attempt,
            )
            if proc.returncode == 0:
                break

            stderr = proc.stderr or ""
            is_apply = len(args) >= 1 and args[0] == "apply"
            validation_timeout = "failed to download openapi" in stderr or "error validating data" in stderr
            transient_timeout_markers = (
                "context deadline exceeded",
                "request canceled",
                "i/o timeout",
                "tls handshake timeout",
                "connection reset by peer",
                "the server is currently unable to handle the request",
            )
            transient_timeout = any(marker in stderr.lower() for marker in transient_timeout_markers)

            if is_apply and validation_timeout and "--validate=false" not in run_args:
                run_args = [args[0], "--validate=false", *args[1:]]
                time.sleep(min(2 * attempt, 5))
                continue

            if transient_timeout and attempt < attempts:
                time.sleep(min(2 * attempt, 5))
                continue

            break

        assert proc is not None
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"kubectl command failed with exit code {proc.returncode}\n"
                f"Args: {' '.join(run_args)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return proc

    def _run_kubectl_exec(self, command: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
        if not self.container_name:
            raise RuntimeError("Container is not initialized")
        return self._run_kubectl(
            ["exec", self.container_name, "--", "bash", "-lc", command],
            check=check,
            timeout=timeout,
        )

    def _sync_k8s_task_tests(self) -> None:
        if not self.container_name:
            raise RuntimeError("Container is not initialized")

        remote_tests_dir = self._task_file("tests")
        if os.path.isdir(remote_tests_dir):
            source_cmd = f"tar -C {shlex.quote(remote_tests_dir)} -cf - ."
        else:
            source_cmd = " ".join(
                shlex.quote(part) for part in [*self._ssh_base_command(), f"tar -C {shlex.quote(remote_tests_dir)} -cf - ."]
            )
        kubectl_cmd = " ".join(
            shlex.quote(part)
            for part in [
                *self._kubectl_base_command(),
                "exec",
                "-i",
                self.container_name,
                "--",
                "bash",
                "-lc",
                "mkdir -p /tests && tar -xf - -C /tests",
            ]
        )
        pipeline = f"set -o pipefail; {source_cmd} | {kubectl_cmd}"

        self._log("kubectl:sync_tests:start", container_name=self.container_name, remote_tests_dir=remote_tests_dir)
        env = os.environ.copy()
        default_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        current_path = env.get("PATH", "")
        env["PATH"] = f"{default_path}:{current_path}" if current_path else default_path
        attempts = 3
        proc = None
        for attempt in range(1, attempts + 1):
            try:
                with self._kube_control_gate:
                    proc = subprocess.run(
                        ["/bin/bash", "-c", pipeline],
                        capture_output=True,
                        text=True,
                        timeout=max(self.step_timeout, 120),
                        env=env,
                    )
            except (BlockingIOError, OSError, RuntimeError) as exc:
                error_text = str(exc)
                self._log("kubectl:sync_tests:spawn_error", error=error_text, attempt=attempt)
                transient_spawn_error = "can't start new thread" in error_text or "Resource temporarily unavailable" in error_text
                if transient_spawn_error and attempt < attempts:
                    time.sleep(min(2 * attempt, 5))
                    continue
                raise
            if proc.returncode == 0 or attempt == attempts:
                break
            time.sleep(min(2 * attempt, 5))
        assert proc is not None
        self._log(
            "kubectl:sync_tests:end",
            returncode=proc.returncode,
            stdout=_truncate(proc.stdout, 300),
            stderr=_truncate(proc.stderr, 300),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to sync task tests into pod {self.container_name}\n"
                f"Remote tests dir: {remote_tests_dir}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

    def _sync_k8s_task_seeds(self) -> None:
        """Sync seeds/ from task directory into pod /workspace/ (for OpenThoughts tasks)."""
        if not self.container_name:
            raise RuntimeError("Container is not initialized")
        seeds_dir = self._task_file("environment/seeds")
        if os.path.isdir(seeds_dir):
            source_args = ["tar", "-C", seeds_dir, "-cf", "-", "."]
        else:
            check_result = self._run_remote(f"test -d {shlex.quote(seeds_dir)} && echo exists", check=False, timeout=30)
            if "exists" not in (check_result.stdout or ""):
                self._log("kubectl:sync_seeds:skip", reason="no seeds dir", seeds_dir=seeds_dir)
                return
            source_args = [*self._ssh_base_command(), f"tar -C {shlex.quote(seeds_dir)} -cf - ."]
        dest_args = [
            *self._kubectl_base_command(),
            "exec",
            "-i",
            self.container_name,
            "--",
            "bash",
            "-lc",
            "mkdir -p /workspace && tar -xf - -C /workspace",
        ]
        env = os.environ.copy()
        default_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        current_path = env.get("PATH", "")
        env["PATH"] = f"{default_path}:{current_path}" if current_path else default_path
        attempts = 3
        for attempt in range(1, attempts + 1):
            self._log("kubectl:sync_seeds:start", container_name=self.container_name, seeds_dir=seeds_dir, attempt=attempt)
            source_proc = None
            dest_proc = None
            try:
                with self._kube_control_gate:
                    source_proc = subprocess.Popen(source_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                    assert source_proc.stdout is not None
                    dest_proc = subprocess.Popen(dest_args, stdin=source_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
                    source_proc.stdout.close()
                    _, dest_stderr = dest_proc.communicate(timeout=120)
                    _, source_stderr = source_proc.communicate(timeout=30)
            except (BlockingIOError, OSError, RuntimeError) as exc:
                error_text = str(exc)
                self._log("kubectl:sync_seeds:spawn_error", error=error_text, attempt=attempt)
                transient_spawn_error = "can't start new thread" in error_text or "Resource temporarily unavailable" in error_text
                if source_proc is not None and source_proc.poll() is None:
                    source_proc.kill()
                    source_proc.wait()
                if dest_proc is not None and dest_proc.poll() is None:
                    dest_proc.kill()
                    dest_proc.wait()
                if transient_spawn_error and attempt < attempts:
                    time.sleep(min(2 * attempt, 5))
                    continue
                raise
            except subprocess.TimeoutExpired:
                if dest_proc is not None and dest_proc.poll() is None:
                    dest_proc.kill()
                    dest_proc.wait()
                if source_proc is not None and source_proc.poll() is None:
                    source_proc.kill()
                    source_proc.wait()
                self._log("kubectl:sync_seeds:warn", msg="seeds sync timed out (non-fatal)")
                return

            dest_stderr_text = dest_stderr.decode("utf-8", errors="replace") if dest_stderr else ""
            source_stderr_text = source_stderr.decode("utf-8", errors="replace") if source_stderr else ""
            source_returncode = source_proc.returncode if source_proc is not None else None
            dest_returncode = dest_proc.returncode if dest_proc is not None else None
            combined_stderr = "\n".join(part for part in [dest_stderr_text, source_stderr_text] if part)
            self._log(
                "kubectl:sync_seeds:end",
                returncode=dest_returncode,
                source_returncode=source_returncode,
                stderr=_truncate(combined_stderr, 300),
            )
            if dest_returncode == 0 and source_returncode == 0:
                return
            if attempt < attempts and "Resource temporarily unavailable" in combined_stderr:
                time.sleep(min(2 * attempt, 5))
                continue
            self._log("kubectl:sync_seeds:warn", msg=f"seeds sync failed (non-fatal): {combined_stderr[:200]}")
            return

    def _ensure_k8s_task_dirs(self) -> None:
        """Create /output/ and /logs/verifier/ inside the pod (needed by OpenThoughts test.sh)."""
        if not self.container_name:
            return
        cmd = "mkdir -p /output /logs/verifier"
        try:
            self._run_kubectl(
                ["exec", self.container_name, "--", "bash", "-lc", cmd],
                check=False, timeout=30,
            )
        except Exception:
            pass

    def _resolve_kube_image(self, docker_image: str) -> str:
        image_map = {}
        if self.kube_image_map:
            for item in self.kube_image_map.split(','):
                entry = item.strip()
                if not entry or '=' not in entry:
                    continue
                src, dst = entry.split('=', 1)
                image_map[src.strip()] = dst.strip()
        if docker_image in image_map:
            return image_map[docker_image]
        if not self.kube_image_registry:
            return docker_image

        registry = self.kube_image_registry.rstrip('/')
        first = docker_image.split('/', 1)[0]
        has_registry = '.' in first or ':' in first or first == 'localhost'
        suffix = docker_image.split('/', 1)[1] if has_registry and '/' in docker_image else docker_image
        return f"{registry}/{suffix}"

    def _ensure_kube_namespace(self) -> None:
        if not os.path.exists(self.kubeconfig_path):
            raise RuntimeError(f"kubeconfig not found at {self.kubeconfig_path}")
        if self.kube_namespace in self._kube_namespace_cache:
            return
        with self._kube_namespace_lock:
            if self.kube_namespace in self._kube_namespace_cache:
                return
            namespace_check = self._run_kubectl(["get", "namespace", self.kube_namespace], check=False, timeout=30)
            if namespace_check.returncode != 0:
                create_proc = self._run_kubectl(["create", "namespace", self.kube_namespace], check=False, timeout=30)
                if create_proc.returncode != 0 and "AlreadyExists" not in (create_proc.stderr or ""):
                    raise RuntimeError(
                        f"failed to ensure namespace {self.kube_namespace}\n"
                        f"STDOUT:\n{create_proc.stdout}\n"
                        f"STDERR:\n{create_proc.stderr}"
                    )
            self._kube_namespace_cache.add(self.kube_namespace)

    def _build_k8s_pod_manifest(
        self,
        docker_image: str,
        env_config: dict[str, Any],
        k8s_services: list[dict[str, Any]] | None = None,
        task_env: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        merged_env = dict(self.container_env)
        for key, value in (task_env or {}).items():
            merged_env[str(key)] = str(value)
        cpu_value = str(env_config.get("cpus", 1))
        memory_value = str(env_config.get("memory", "4Gi"))
        cpu_request_value = str(env_config.get("cpu_request", "100m"))
        memory_request_value = str(env_config.get("memory_request", memory_value))
        containers = []
        hostnames = set()
        for service in k8s_services or []:
            service_name = str(service["name"])
            service_image = self._resolve_kube_image(str(service["image"]))
            service_env = [{"name": key, "value": str(value)} for key, value in service.get("env", {}).items()]
            service_cpu = str(service.get("cpu", "250m"))
            service_memory = str(service.get("memory", "512Mi"))
            service_cpu_request = str(service.get("cpu_request", "50m"))
            service_memory_request = str(service.get("memory_request", service_memory))
            container = {
                "name": service_name,
                "image": service_image,
                "imagePullPolicy": self.kube_image_pull_policy,
                "env": service_env,
                "resources": {
                    "requests": {"cpu": service_cpu_request, "memory": service_memory_request},
                    "limits": {"cpu": service_cpu, "memory": service_memory},
                },
            }
            if service.get("working_dir"):
                container["workingDir"] = str(service["working_dir"])
            if service.get("command"):
                container["command"] = [str(item) for item in service["command"]]
            if service.get("args"):
                container["args"] = [str(item) for item in service["args"]]
            if service.get("ports"):
                container["ports"] = [{"containerPort": int(port)} for port in service["ports"]]
            if service.get("readiness_probe"):
                container["readinessProbe"] = service["readiness_probe"]
            if service.get("startup_probe"):
                container["startupProbe"] = service["startup_probe"]
            if service.get("liveness_probe"):
                container["livenessProbe"] = service["liveness_probe"]
            containers.append(container)
            hostnames.add(service_name)
            for alias in service.get("aliases", []):
                hostnames.add(str(alias))
        if hostnames:
            no_proxy_keys = ("NO_PROXY", "no_proxy")
            for key in no_proxy_keys:
                current = merged_env.get(key, "")
                pieces = [item.strip() for item in str(current).split(",") if item.strip()]
                for hostname in sorted(hostnames):
                    if hostname not in pieces:
                        pieces.append(hostname)
                merged_env[key] = ",".join(pieces)
        env_items = [{"name": key, "value": value} for key, value in merged_env.items() if value]
        containers.insert(
            0,
            {
                "name": "task",
                "image": str(docker_image),
                "imagePullPolicy": self.kube_image_pull_policy,
                "workingDir": "/app",
                "command": ["bash", "-lc", "trap 'exit 0' TERM INT; while true; do sleep 3600; done"],
                "env": env_items,
                "resources": {
                    "requests": {"cpu": cpu_request_value, "memory": memory_request_value},
                    "limits": {"cpu": cpu_value, "memory": memory_value},
                },
                "volumeMounts": [
                    {"name": "tests", "mountPath": "/tests"},
                    {"name": "verifier-logs", "mountPath": "/logs/verifier"},
                ],
            },
        )

        spec = {
            "restartPolicy": "Always" if k8s_services else "Never",
            "containers": containers,
            "volumes": [
                {
                    "name": "tests",
                    "emptyDir": {},
                },
                {
                    "name": "verifier-logs",
                    "emptyDir": {},
                },
            ],
        }
        if hostnames:
            spec["hostAliases"] = [{"ip": "127.0.0.1", "hostnames": sorted(hostnames)}]
        if self.kube_node_name:
            spec["nodeName"] = self.kube_node_name

        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self.container_name,
                "namespace": self.kube_namespace,
                "labels": {
                    "app.kubernetes.io/name": "terminal-bench-direct",
                    "tb-task-name": self.task_name or "unknown",
                },
            },
            "spec": spec,
        }

    def _build_ssh_control_path(self) -> str:
        base_dir = os.environ.get("TB_SSH_CONTROL_DIR", "/tmp/rllm-tb-ssh")
        fingerprint = hashlib.sha1(
            f"{self.harbor_user}@{self.harbor_host}:{self.harbor_port}|{self.harbor_ssh_key}".encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(base_dir, f"cm-{fingerprint}")

    def _ssh_base_command(self) -> list[str]:
        control_dir = os.path.dirname(self._ssh_control_path)
        if control_dir:
            os.makedirs(control_dir, exist_ok=True)
        ssh_bin = os.environ.get("TB_SSH_BIN") or shutil.which("ssh") or "/usr/bin/ssh"
        return [
            ssh_bin,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=600",
            "-o",
            f"ControlPath={self._ssh_control_path}",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ConnectionAttempts=3",
            "-p",
            str(self.harbor_port),
            "-i",
            self.harbor_ssh_key,
            f"{self.harbor_user}@{self.harbor_host}",
        ]

    def _run_remote(self, command: str, check: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
        self._log("remote:start", timeout=timeout, command=_truncate(command, 500))
        attempts = 3
        proc = None
        for attempt in range(1, attempts + 1):
            proc = subprocess.run(
                [*self._ssh_base_command(), command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            self._log(
                "remote:attempt",
                attempt=attempt,
                returncode=proc.returncode,
                stdout=_truncate(proc.stdout, 300),
                stderr=_truncate(proc.stderr, 300),
            )
            stderr = proc.stderr or ""
            should_retry = proc.returncode == 255 or "Connection reset by" in stderr or "kex_exchange_identification" in stderr
            if not should_retry or attempt == attempts:
                break
            time.sleep(min(2 * attempt, 5))
        assert proc is not None
        self._log("remote:end", returncode=proc.returncode, stdout=_truncate(proc.stdout, 300), stderr=_truncate(proc.stderr, 300))
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Remote command failed with exit code {proc.returncode}\n"
                f"Command: {command}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return proc

    def _log(self, event: str, **payload: Any) -> None:
        if not self.debug_log_path:
            return
        log_dir = os.path.dirname(self.debug_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "task_name": self.task_name,
            "container_name": self.container_name,
            **payload,
        }
        with open(self.debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _default_container_env() -> dict[str, str]:
    proxy = os.environ.get("TB_TASK_HTTP_PROXY") or os.environ.get("HTTP_PROXY") or "http://100.66.21.88:3128"
    no_proxy = (
        os.environ.get("TB_TASK_NO_PROXY")
        or os.environ.get("NO_PROXY")
        or "localhost,127.0.0.1,.svc,.cluster.local,vke-cn-shanghai.cr.volces.com,tck-xinan-registry.cn-chengdu.cr.aliyuncs.com"
    )
    return {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    }


def _with_line_numbers(content: str) -> str:
    lines = content.splitlines()
    if not lines:
        return "1\t"
    return "\n".join(f"{idx}\t{line}" for idx, line in enumerate(lines, start=1))


def _python_write_file_script(path: str, encoded: str) -> str:
    return (
        "import base64\n"
        "from pathlib import Path\n"
        f"path = Path({path!r})\n"
        f"content = base64.b64decode({encoded!r}).decode('utf-8')\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text(content, encoding='utf-8')\n"
    )


def _truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"
