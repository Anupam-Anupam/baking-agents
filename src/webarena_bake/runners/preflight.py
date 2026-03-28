from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from webarena_bake.utils.io import read_json


REQUIRED_WEBARENA_ENV_VARS = (
    "SHOPPING",
    "SHOPPING_ADMIN",
    "REDDIT",
    "GITLAB",
    "MAP",
    "WIKIPEDIA",
    "HOMEPAGE",
)

DEFAULT_SITE_ENV = {
    "SHOPPING": "http://onestopmarket.com",
    "SHOPPING_ADMIN": "http://luma.com/admin",
    "REDDIT": "http://reddit.com",
    "GITLAB": "http://gitlab.com",
    "MAP": "http://openstreetmap.org",
    "WIKIPEDIA": "https://en.wikipedia.org/wiki/Main_Page",
    "HOMEPAGE": "https://webarena.dev",
}


def _normalize_url(value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        return value
    return f"http://{value}"


def _extract_host_port(value: str) -> tuple[str, int | None]:
    normalized = _normalize_url(value)
    parsed = urlparse(normalized)
    return parsed.hostname or "", parsed.port


def _probe_url(url: str, timeout: float = 2.5) -> dict[str, Any]:
    try:
        response = requests.get(_normalize_url(url), timeout=timeout)
        return {"ok": response.status_code < 500, "status_code": response.status_code, "error": ""}
    except Exception as exc:  # pragma: no cover - diagnostics path
        return {"ok": False, "status_code": 0, "error": str(exc)}


def _probe_models_endpoint(endpoint: str, timeout: float = 2.5) -> dict[str, Any]:
    url = f"{endpoint.rstrip('/')}/models"
    try:
        response = requests.get(url, timeout=timeout)
        model_ids: list[str] = []
        try:
            payload = response.json()
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            raw_data = payload.get("data", [])
            if isinstance(raw_data, list):
                for item in raw_data:
                    if isinstance(item, dict) and "id" in item:
                        model_ids.append(str(item["id"]))
        return {
            "url": url,
            "ok": response.status_code < 500,
            "status_code": response.status_code,
            "error": "",
            "model_ids": model_ids,
            "server": response.headers.get("Server", ""),
        }
    except Exception as exc:  # pragma: no cover - diagnostics path
        return {
            "url": url,
            "ok": False,
            "status_code": 0,
            "error": str(exc),
            "model_ids": [],
            "server": "",
        }


def _probe_tcp(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _docker_ready() -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return True, ""
        err = proc.stderr.strip() or proc.stdout.strip() or "docker_info_failed"
        return False, err
    except FileNotFoundError:
        return False, "docker_not_installed_or_not_in_path"


def collect_required_auth_files(
    webarena_root: Path,
    webarena_config_dir: Path,
    task_ids: list[str],
) -> list[Path]:
    auth_files: list[Path] = []
    for task_id in task_ids:
        cfg_path = webarena_config_dir / f"{task_id}.json"
        if not cfg_path.exists():
            continue
        payload = read_json(cfg_path)
        if not isinstance(payload, dict):
            continue
        storage_state = payload.get("storage_state")
        if not storage_state:
            continue
        storage_state_str = str(storage_state)
        relative_state = (
            storage_state_str[2:] if storage_state_str.startswith("./") else storage_state_str
        )
        state_path = webarena_root / relative_state
        auth_files.append(state_path)
    unique = sorted(set(auth_files))
    return unique


def _collect_required_service_urls(
    webarena_config_dir: Path,
    task_ids: list[str],
    env_checks: dict[str, dict[str, Any]],
) -> list[str]:
    urls: list[str] = []
    site_to_env = {
        "shopping": "SHOPPING",
        "shopping_admin": "SHOPPING_ADMIN",
        "reddit": "REDDIT",
        "gitlab": "GITLAB",
        "map": "MAP",
        "wikipedia": "WIKIPEDIA",
        "homepage": "HOMEPAGE",
    }
    for task_id in task_ids:
        cfg_path = webarena_config_dir / f"{task_id}.json"
        if not cfg_path.exists():
            continue
        payload = read_json(cfg_path)
        if not isinstance(payload, dict):
            continue
        start_url = str(payload.get("start_url", "")).strip()
        if start_url.startswith("localhost"):
            for site in payload.get("sites", []):
                env_key = site_to_env.get(str(site))
                if env_key and env_key in env_checks:
                    effective = str(env_checks[env_key]["effective_value"]).strip()
                    if effective:
                        urls.append(effective)
        if start_url:
            urls.append(start_url)
    return sorted(set(urls))


def run_preflight(
    webarena_root: Path,
    webarena_config_dir: Path,
    task_ids: list[str],
    policy_model_endpoint: str,
    observer_model_endpoint: str | None,
    provider: str,
    required_env_vars: tuple[str, ...] = REQUIRED_WEBARENA_ENV_VARS,
) -> dict[str, Any]:
    docker_ok, docker_error = _docker_ready()

    env_checks: dict[str, dict[str, Any]] = {}
    for name in required_env_vars:
        value = os.environ.get(name, "")
        default_value = DEFAULT_SITE_ENV.get(name, "")
        effective_value = value or default_value
        env_checks[name] = {
            "set": bool(effective_value),
            "value": value,
            "default_value": default_value,
            "effective_value": effective_value,
        }

    required_auth_files = collect_required_auth_files(webarena_root, webarena_config_dir, task_ids)
    auth_checks = [{"path": str(path), "exists": path.exists()} for path in required_auth_files]
    missing_auth = [item["path"] for item in auth_checks if not item["exists"]]

    service_urls = _collect_required_service_urls(webarena_config_dir, task_ids, env_checks)
    service_checks: list[dict[str, Any]] = []
    for url in service_urls:
        host, port = _extract_host_port(url)
        tcp_ok = bool(host and port and _probe_tcp(host, port))
        http_probe = _probe_url(url)
        service_checks.append(
            {
                "url": _normalize_url(url),
                "host": host,
                "port": port,
                "tcp_ok": tcp_ok,
                "http_ok": bool(http_probe["ok"]),
                "http_status": http_probe["status_code"],
                "http_error": http_probe["error"],
            }
        )

    provider_checks: dict[str, Any] = {"provider": provider}
    if provider in {"openai", "openai_compatible"}:
        provider_checks["OPENAI_API_KEY_set"] = bool(os.environ.get("OPENAI_API_KEY"))
    policy_health = _probe_models_endpoint(policy_model_endpoint)
    policy_model_ids = [item.lower() for item in policy_health.get("model_ids", [])]
    policy_server = str(policy_health.get("server", "")).lower()
    policy_looks_like_mock = (
        "mockopenaiserver" in policy_server
        or any(item.startswith("mock") or "mock-qwen" in item for item in policy_model_ids)
    )
    provider_checks["policy_endpoint"] = {
        **policy_health,
        "looks_like_mock": policy_looks_like_mock,
    }
    if observer_model_endpoint:
        observer_health = _probe_models_endpoint(observer_model_endpoint)
        provider_checks["observer_endpoint"] = {
            **observer_health,
        }

    all_env_set = all(item["set"] for item in env_checks.values())
    all_services_up = all(item["tcp_ok"] or item["http_ok"] for item in service_checks) if service_checks else False
    requires_local_docker = any(item.get("host") in {"localhost", "127.0.0.1"} for item in service_checks)
    policy_up = bool(provider_checks["policy_endpoint"]["ok"])
    observer_up = True
    if observer_model_endpoint:
        observer_up = bool(provider_checks["observer_endpoint"]["ok"])
    policy_not_mock = not policy_looks_like_mock

    docker_gate_ok = docker_ok if requires_local_docker else True
    auth_gate_ok = (not missing_auth) if requires_local_docker else True
    ok = docker_gate_ok and all_env_set and auth_gate_ok and all_services_up and policy_up and observer_up and policy_not_mock
    hints: list[str] = []
    if requires_local_docker and not docker_ok:
        hints.append(f"Docker daemon unavailable: {docker_error}")
    if not all_env_set:
        missing_vars = [name for name, item in env_checks.items() if not item["set"]]
        hints.append(f"Missing required env vars: {', '.join(missing_vars)}")
    if missing_auth and requires_local_docker:
        hints.append("Missing storage_state auth files. Run browser_env/auto_login.py once services are up.")
    if not all_services_up:
        hints.append("One or more WebArena site URLs are unreachable. Start website services and re-run preflight.")
    if not policy_up:
        hints.append("Policy model endpoint is unreachable. Start your model server or update policy_model_endpoint.")
    if policy_looks_like_mock:
        hints.append(
            "Policy endpoint appears to be scripts/mock_openai_server.py (mock-qwen). "
            "It returns canned 'stop [N/A]' responses and will make most tasks fail. "
            "Use a real model-serving endpoint before training "
            "(for example: scripts/tinker_openai_server.py)."
        )
    if observer_model_endpoint and not observer_up:
        hints.append("Observer model endpoint is unreachable. Start observer server or disable observer endpoint.")

    return {
        "ok": ok,
        "requires_local_docker": requires_local_docker,
        "docker": {"ok": docker_ok, "error": docker_error},
        "env": env_checks,
        "auth": {"files": auth_checks, "missing_count": len(missing_auth)},
        "services": service_checks,
        "provider": provider_checks,
        "hints": hints,
    }
