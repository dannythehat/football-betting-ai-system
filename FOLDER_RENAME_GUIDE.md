# Folder Rename Guide

## Problem
The repository has folders with hyphens (`data-ingestion`, `smart-bets-ai`, etc.) but Python imports use underscores (`data_ingestion`, `smart_bets_ai`). This causes import failures.

## Solution
Rename folders to match Python import conventions (underscores instead of hyphens).

## Required Folder Renames

| Current Name | New Name |
|--------------|----------|
| `data-ingestion` | `data_ingestion` |
| `smart-bets-ai` | `smart_bets_ai` |
| `golden-bets-ai` | `golden_bets_ai` |
| `value-bets-ai` | `value_bets_ai` |
| `custom-analysis` | `custom_analysis` |

## How to Rename (Local Git)

Since GitHub API doesn't support direct folder renaming, you need to do this locally:

```bash
# Clone the repository
git clone https://github.com/dannythehat/football-betting-ai-system.git
cd football-betting-ai-system

# Checkout the fix branch
git checkout fix/folder-renames-and-imports

# Rename folders using git mv
git mv data-ingestion data_ingestion
git mv smart-bets-ai smart_bets_ai
git mv golden-bets-ai golden_bets_ai
git mv value-bets-ai value_bets_ai
git mv custom-analysis custom_analysis

# Commit the changes
git commit -m "Rename folders to use underscores for Python imports"

# Push to GitHub
git push origin fix/folder-renames-and-imports
```

## What's Already Fixed

✅ **Import fix in smart_bets_ai/predict.py**
- Changed `from features import FeatureEngineer` to `from .features import FeatureEngineer`

✅ **Health endpoint**
- `/health` endpoint already exists in user-api/main.py

✅ **Models directory**
- Created `models/` directory at repo root with `.gitkeep`

## After Renaming

Once folders are renamed, the following will work:

```python
# These imports will now succeed
from data_ingestion.database import get_db_session
from smart_bets_ai.predict import SmartBetsPredictor
from golden_bets_ai.predict import GoldenBetsPredictor
from value_bets_ai.predict import ValueBetsPredictor
from custom_analysis import CustomBetAnalyzer
```

## Testing Locally

After renaming, test that the API starts successfully:

```bash
cd user-api
uvicorn main:app --host 127.0.0.1 --port 8000
```

Then verify endpoints:
- http://127.0.0.1:8000/
- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/docs

## Deployment to Render

After local testing succeeds, Render deployment should work automatically since `render.yaml` is already configured correctly.

## Files Modified in This Branch

1. `smart-bets-ai/predict.py` - Fixed import to use relative import
2. `models/.gitkeep` - Created models directory
3. `scripts/fix_folder_names.py` - Documentation script
4. `FOLDER_RENAME_GUIDE.md` - This guide

## Next Steps

1. Complete folder renames locally (see commands above)
2. Test locally with `uvicorn`
3. Merge PR to main
4. Deploy to Render
5. Train models (optional but recommended)
