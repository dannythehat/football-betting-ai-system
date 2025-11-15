#!/usr/bin/env python3
"""
Script to document folder rename operations needed for proper Python imports.

This script documents the manual steps needed since GitHub API doesn't support
direct folder renaming. These operations should be done via git commands:

git mv data-ingestion data_ingestion
git mv smart-bets-ai smart_bets_ai
git mv golden-bets-ai golden_bets_ai
git mv value-bets-ai value_bets_ai
git mv custom-analysis custom_analysis
git commit -m "Rename folders to use underscores for Python imports"
git push

After renaming, the following import fix is needed:
- smart_bets_ai/predict.py line 12: from features -> from .features
"""

import os
import sys

RENAMES = {
    'data-ingestion': 'data_ingestion',
    'smart-bets-ai': 'smart_bets_ai',
    'golden-bets-ai': 'golden_bets_ai',
    'value-bets-ai': 'value_bets_ai',
    'custom-analysis': 'custom_analysis'
}

def main():
    print("Folder Rename Operations Needed:")
    print("=" * 50)
    for old, new in RENAMES.items():
        print(f"  {old} → {new}")
    print("\nRun these git commands:")
    for old, new in RENAMES.items():
        print(f"  git mv {old} {new}")
    print("\nThen commit and push:")
    print("  git commit -m 'Rename folders to use underscores'")
    print("  git push")

if __name__ == '__main__':
    main()
