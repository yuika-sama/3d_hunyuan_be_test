#!/usr/bin/env python3
"""
Simple zero-dependency HTTP server & CORS proxy for Hunyuan3D Test UI.
Usage:
    python serve_test_ui.py [port]
Default port: 8080
"""
import http.server
import json
import os
import sys
import urllib.request
import urllib.error
import webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestUIRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=CURRENT_DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/proxy/"):
            self.handle_runpod_proxy()
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/test_ui.html"
            return super().do_GET()
        else:
            return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/proxy/"):
            self.handle_runpod_proxy()
        else:
            self.send_error(404, "Not Found")

    def handle_runpod_proxy(self):
        # Maps /proxy/<endpoint_id>/<action> -> https://api.runpod.ai/v2/<endpoint_id>/<action>
        subpath = self.path[len("/proxy/") :]
        target_url = f"https://api.runpod.ai/v2/{subpath}"

        headers = {}
        for key, val in self.headers.items():
            if key.lower() in ("authorization", "content-type", "accept"):
                headers[key] = val

        body = None
        if "content-length" in self.headers:
            length = int(self.headers["content-length"])
            body = self.rfile.read(length)

        req = urllib.request.Request(
            target_url, data=body, headers=headers, method=self.command
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp_data = resp.read()
                self.send_response(resp.status)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(resp_data)
        except urllib.error.HTTPError as e:
            err_data = e.read()
            self.send_response(e.code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(err_data)
        except Exception as e:
            self.send_response(500)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())


def run_server():
    server_address = ("", PORT)
    httpd = http.server.HTTPServer(server_address, TestUIRequestHandler)
    url = f"http://localhost:{PORT}"
    print("=" * 60)
    print(f"🚀 Hunyuan3D Serverless Test Studio is running!")
    print(f"👉 Local URL: {url}")
    print(f"👉 File Path: {os.path.join(CURRENT_DIR, 'test_ui.html')}")
    print("   (Press Ctrl+C to stop)")
    print("=" * 60)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    run_server()
