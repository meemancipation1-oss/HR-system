"""Launch the FastAPI server."""
from __future__ import annotations
import uvicorn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    print("Starting HR Multi-Agent API server...")
    print(f"   Docs:  http://localhost:8000/docs")
    print(f"   Root:  http://localhost:8000/")
    uvicorn.run(
        "api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
