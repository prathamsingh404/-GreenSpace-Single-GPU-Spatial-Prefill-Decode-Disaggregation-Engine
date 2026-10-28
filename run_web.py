"""
Quickstart launcher for the Project GreenSpace Web Control Center.
Runs the FastAPI web server with Uvicorn and automatically launches the user's default browser.
"""

import os
import sys
import webbrowser
import threading
import time
import uvicorn

def open_browser():
    time.sleep(1.2)
    url = "http://localhost:8000"
    print(f"\n>>> Opening Project GreenSpace Web Control Center in your browser at {url}...\n")
    webbrowser.open(url)

if __name__ == "__main__":
    print("=" * 70)
    print("   PROJECT GREENSPACE: SINGLE-GPU SPATIAL PREFILL-DECODE WEB ENGINE")
    print("=" * 70)
    print("Serving on http://localhost:8000 (Ctrl+C to stop)...")
    
    # Launch browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run("greenspace.web.server:app", host="0.0.0.0", port=8000, reload=False, log_level="info")

