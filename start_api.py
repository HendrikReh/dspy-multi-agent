#!/usr/bin/env python3
"""Startup script for the DSPy Multi-Agent FastAPI server."""

import subprocess
import sys
from pathlib import Path


def main():
    """Start the FastAPI server with proper configuration."""
    # Change to project directory
    project_root = Path(__file__).parent
    
    # Command to start the server
    cmd = [
        "uv", "run", "uvicorn", 
        "src.api.main:app",
        "--reload",
        "--host", "0.0.0.0", 
        "--port", "8000"
    ]
    
    print("Starting DSPy Multi-Agent FastAPI server...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 