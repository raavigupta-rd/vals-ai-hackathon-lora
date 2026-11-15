"""Shared Tinker settings for this repo."""

import os

# Tinker Console project that all sessions are grouped under.
# Override with TINKER_PROJECT_ID if needed.
PROJECT_ID = os.environ.get("TINKER_PROJECT_ID", "7fe227d5-664d-42d0-a48b-d37173661803")

BASE_MODEL = "Qwen/Qwen3.5-9B"
