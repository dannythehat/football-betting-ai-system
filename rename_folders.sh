#!/bin/bash
# Automated folder renaming script
# Run this after checking out the fix/folder-renames-and-imports branch

set -e

echo "🔧 Renaming folders to use underscores..."
echo ""

# Check if we're in the right directory
if [ ! -f "README.md" ]; then
    echo "❌ Error: Please run this script from the repository root"
    exit 1
fi

# Check if we're on the right branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "fix/folder-renames-and-imports" ]; then
    echo "⚠️  Warning: You're on branch '$CURRENT_BRANCH'"
    echo "   Recommended to run on 'fix/folder-renames-and-imports' branch"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Rename folders
echo "Renaming: data-ingestion → data_ingestion"
git mv data-ingestion data_ingestion

echo "Renaming: smart-bets-ai → smart_bets_ai"
git mv smart-bets-ai smart_bets_ai

echo "Renaming: golden-bets-ai → golden_bets_ai"
git mv golden-bets-ai golden_bets_ai

echo "Renaming: value-bets-ai → value_bets_ai"
git mv value-bets-ai value_bets_ai

echo "Renaming: custom-analysis → custom_analysis"
git mv custom-analysis custom_analysis

echo ""
echo "✅ All folders renamed successfully!"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Commit: git commit -m 'Rename folders to use underscores'"
echo "3. Push: git push origin fix/folder-renames-and-imports"
echo "4. Test locally: cd user-api && uvicorn main:app --host 127.0.0.1 --port 8000"
