import json
import urllib.request
import urllib.parse
import urllib.error
import yaml
import os
import base64
import time
from typing import Any, Dict, Optional

from mint.tools.base import Tool


def _truncate(s: str, n: int = 2000) -> str:
    if len(s) > n:
        return s[:n] + "\n[Output Truncated]"
    return s


def _sanitize(s: str) -> str:
    # Only replace non-empty secrets to avoid corrupting output when env vars are unset
    for env_name in ("OPENAI_API_KEY", "SPOTIFY_TOKEN"):
        val = os.environ.get(env_name)
        if val and val in s:
            s = s.replace(val, "<hidden>")
    return s


def _read_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class APISpecToolBase(Tool):
    name = "api_tool"
    signature = "api_tool(method_path: str, params: dict=None, path_params: dict=None, body: dict=None) -> str"
    description = "Generic API tool using OpenAPI spec; method_path like 'GET /search/movie'"

    def __init__(self, spec_path: str, server_url: str, auth_header: Optional[str] = None):
        self.spec_path = spec_path
        self.server_url = server_url.rstrip("/")
        self.auth_header = auth_header

    def _build_url(self, path: str, path_params: Optional[Dict[str, Any]] = None, params: Optional[Dict[str, Any]] = None) -> str:
        url = self.server_url + path
        if path_params:
            for k, v in path_params.items():
                url = url.replace("{" + k + "}", urllib.parse.quote(str(v)))
        if params:
            qs = urllib.parse.urlencode(params, doseq=True)
            url += ("?" + qs)
        return url

    def __call__(self, method_path: str, params: Dict[str, Any] = None, path_params: Dict[str, Any] = None, body: Dict[str, Any] = None) -> str:
        try:
            # Support flexible method_path: "GET /path", "/path", or "path"
            if " " in method_path:
                method, path = method_path.split(" ", 1)
            else:
                method, path = "GET", method_path
            method = method.upper().strip()
            path = path.strip()
            if not path.startswith("/"):
                path = "/" + path
            url = self._build_url(path, path_params, params)
            headers = {"Accept": "application/json"}
            if self.auth_header:
                headers.update(self.auth_header)

            headers_safe = dict(headers)
            if "Authorization" in headers_safe:
                headers_safe["Authorization"] = "<hidden>"

            print(f"[api_tool] {method} {url}")
            print(f"[api_tool] headers: {json.dumps(headers_safe)}")
            print(f"[api_tool] path_params: {json.dumps(path_params or {})}")
            print(f"[api_tool] params: {json.dumps(params or {})}")

            data = None
            if body is not None:
                data = json.dumps(body).encode("utf-8")
                headers["Content-Type"] = "application/json"
                try:
                    print(f"[api_tool] body: {json.dumps(body)[:500]}")
                except Exception:
                    print("[api_tool] body: <unserializable>")

            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
                out = _truncate(_sanitize(content))
                print(f"[api_tool] status: {resp.status}")
                return out
        except urllib.error.HTTPError as e:
            try:
                msg = e.read().decode("utf-8", errors="ignore")
            except Exception:
                msg = str(e)
            smsg = _sanitize(msg)
            print(f"[api_tool] HTTPError {e.code}: {smsg[:500]}")
            err = {"error": {"type": "HTTPError", "code": e.code, "message": smsg}}
            return _truncate(json.dumps(err))
        except Exception as e:
            emsg = str(e)
            print(f"[api_tool] NetworkError: {emsg}")
            err = {"error": {"type": "NetworkError", "message": emsg}}
            return _truncate(json.dumps(err))


class TMDBAPITool(APISpecToolBase):
    name = "tmdb_api"
    signature = "tmdb_api(method_path: str, params: dict=None, path_params: dict=None, body: dict=None) -> str"
    description = "Call TMDB Web API using method_path and parameters"

    def __init__(self):
        spec = os.path.join("./experiment/specs", "tmdb_oas.json")
        cfg = _read_config("./config.yaml")
        token = None
        api_section = cfg.get("API Key", {})
        if isinstance(api_section, dict):
            token = api_section.get("tmdb")
        auth = {"Authorization": f"Bearer {token}"} if token else None
        super().__init__(spec_path=spec, server_url="https://api.themoviedb.org/3", auth_header=auth)


class SpotifyAPITool(APISpecToolBase):
    name = "spotify_api"
    signature = "spotify_api(method_path: str, params: dict=None, path_params: dict=None, body: dict=None) -> str"
    description = "Call Spotify Web API using method_path and parameters (requires OAuth access token)"

    def __init__(self):
        spec = os.path.join("./experiment/specs", "spotify_oas.json")
        cfg = _read_config("./config.yaml")
        token = None
        api_section = cfg.get("API Key", {})
        if isinstance(api_section, dict):
            token = api_section.get("spotify")
        client_id = None
        client_secret = None
        redirect_uri = None
        if isinstance(api_section, dict):
            client_id = api_section.get("spotipy_client_id")
            client_secret = api_section.get("spotipy_client_secret")
            redirect_uri = api_section.get("spotipy_redirect_uri")
        if not token:
            token = os.environ.get("SPOTIFY_TOKEN", "")
        if not token and client_id and client_secret:
            token = self._get_client_credentials_token(client_id, client_secret)
        auth = {"Authorization": f"Bearer {token}"} if token else None
        self._client_id = client_id
        self._client_secret = client_secret
        self._last_token_time = time.time()
        super().__init__(spec_path=spec, server_url="https://api.spotify.com/v1", auth_header=auth)

    def _get_client_credentials_token(self, client_id: str, client_secret: str) -> Optional[str]:
        try:
            token_url = "https://accounts.spotify.com/api/token"
            data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
            creds = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
            headers = {"Authorization": f"Basic {creds}", "Content-Type": "application/x-www-form-urlencoded"}
            req = urllib.request.Request(token_url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=15) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
                return payload.get("access_token")
        except Exception:
            return None

    def __call__(self, method_path: str, params: Dict[str, Any] = None, path_params: Dict[str, Any] = None, body: Dict[str, Any] = None) -> str:
        try:
            out = super().__call__(method_path, params=params, path_params=path_params, body=body)
            if out.startswith("HTTPError 401") and self._client_id and self._client_secret:
                new_token = self._get_client_credentials_token(self._client_id, self._client_secret)
                if new_token:
                    self.auth_header = {"Authorization": f"Bearer {new_token}"}
                    out = super().__call__(method_path, params=params, path_params=path_params, body=body)
            return out
        except Exception as e:
            return _truncate(f"ToolError: {e}")

