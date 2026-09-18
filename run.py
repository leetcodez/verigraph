#!/usr/bin/env python3
"""
VeriGraph Universal Launcher
Runs natively across Windows, macOS, and Linux.
Provides automatic port fallback, pre-flight dependency verification, and directory initialization.
"""

import sys
import os
import socket
import argparse
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Minimum Python version
if sys.version_info < (3, 9):
    sys.exit(f"[-] Python 3.9+ is required. Current version: {sys.version}")


def check_dependencies() -> bool:
    """Verify that essential packages are installed."""
    required = ["fastapi", "uvicorn", "pydantic", "networkx", "numpy", "pydantic_settings"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print("\n" + "=" * 65)
        print(" [!] MISSING DEPENDENCIES DETECTED")
        print("=" * 65)
        print(f"Missing packages: {', '.join(missing)}")
        print("\nPlease install the project dependencies by running:")
        print("    pip install -r requirements.txt\n")
        print("=" * 65 + "\n")
        return False
    return True


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is currently bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_available_port(start_port: int, host: str = "127.0.0.1", max_attempts: int = 20) -> int:
    """Find the first available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port, host):
            return port
    return start_port


def main():
    parser = argparse.ArgumentParser(description="VeriGraph Enterprise Engine Launcher")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"), help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")), help="Port to bind (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload on code change")
    args = parser.parse_args()

    # Pre-flight check
    if not check_dependencies():
        sys.exit(1)

    # Initialize data directories
    for sub in ["storage", "uploads", "benchmarks"]:
        (PROJECT_ROOT / "data" / sub).mkdir(parents=True, exist_ok=True)

    # Determine port (with graceful fallback if default 8080 is occupied)
    requested_port = args.port
    selected_port = find_available_port(requested_port, "127.0.0.1")
    if selected_port != requested_port:
        print(f"[*] Port {requested_port} is in use; automatically bound to available port {selected_port}")

    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    dash_url = f"http://{display_host}:{selected_port}/"
    docs_url = f"http://{display_host}:{selected_port}/docs"

    banner = f"""
========================================================================
   VeriGraph: Enterprise Agentic GraphRAG & Verifiable Retrieval Engine
========================================================================
[*] Application Root:       {PROJECT_ROOT}
[*] Interactive Dashboard:  {dash_url}
[*] OpenAPI Swagger Docs:  {docs_url}
[*] Status:                 Ready. Press Ctrl+C to stop.
========================================================================
"""
    print(banner)

    # Try opening browser if in interactive desktop session
    if not args.no_browser and os.getenv("DISPLAY") or sys.platform in ("win32", "darwin"):
        try:
            webbrowser.open(dash_url)
        except Exception:
            pass

    import uvicorn
    try:
        uvicorn.run(
            "verigraph.api.app:app",
            host=args.host,
            port=selected_port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[*] VeriGraph gracefully shut down.")


if __name__ == "__main__":
    main()
