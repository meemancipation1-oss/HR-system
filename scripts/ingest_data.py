"""Script to ingest and build the vector index from HR CSV data."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.rag import RAGEngine


def main():
    print("=" * 60)
    print("HR Data Ingestion — Building Vector Index")
    print("=" * 60)

    engine = RAGEngine()
    print("\n[1/2] Reading CSV files and building documents...")
    engine.build_index(force=True)

    print("\n[2/2] Index built successfully!")
    print("\nReady for semantic search and RAG queries.")


if __name__ == "__main__":
    main()
