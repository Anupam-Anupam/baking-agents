"""
Bread SDK - Bake Yoda Personality Example

Before running this script:
1. Install the SDK: pip install aibread
2. Set your API key: export BREAD_API_KEY='your_api_key_here'
   - Windows CMD: setx BREAD_API_KEY "your_api_key_here"
   - Windows PowerShell: $env:BREAD_API_KEY='your_api_key_here'
"""

import os
from aibread import Bread

# ============ CONFIGURATION ============
REPO_NAME = "yoda_repo"
TARGET_NAME = "yoda_target"
BAKE_NAME = "yoda_bake_v1"
# =======================================

# Initialize client
client = Bread(api_key=os.environ.get("BREAD_API_KEY"))

# Create repository
client.repo.set(repo_name=REPO_NAME)

# Create teacher prompt (u) - The Yoda personality to bake in
client.prompts.set(
    prompt_name="yoda_teacher",
    repo_name=REPO_NAME,
    messages=[{
        "role": "system",
        "content": "You are Yoda. Speak like Yoda, use inverted syntax, few words, and wise, cryptic tone, always calm and reflective."
    }]
)

# Create student prompt (v) - Empty for always-on behavior
client.prompts.set(
    prompt_name="empty_student",
    repo_name=REPO_NAME,
    messages=[{
        "role": "system",
        "content": ""  # Empty = model ALWAYS acts like Yoda
    }]
)

# Configure target
client.targets.set(
    target_name=TARGET_NAME,
    repo_name=REPO_NAME,
    template="default",
    overrides={
        "generators": [
            {
                "type": "hardcoded",
                "numq": 4,
                "questions": [
                    "How can I find balance in the Force?",
                    "Hello, this is Anakin Skywalker",
                    "How tall are you?",
                    "Teach me about patience."
                ]
            },
            {
                "type": "oneshot_qs",
                "model": "claude-sonnet-4-5-20250929",
                "numq": 100,
                "temperature": 0.6
            }
        ],
        "u": "yoda_teacher",    # Teacher: Yoda personality
        "v": "empty_student"    # Student: empty (always-on)
    }
)

# Run stim (generate questions)
client.targets.stim.run(target_name=TARGET_NAME, repo_name=REPO_NAME)

# Run rollout (generate Yoda's responses)
client.targets.rollout.run(target_name=TARGET_NAME, repo_name=REPO_NAME)

# Configure bake
client.bakes.set(
    bake_name=BAKE_NAME,
    repo_name=REPO_NAME,
    template="default",
    overrides={
        "datasets": [{"target": TARGET_NAME, "weight": 1.0}]
    }
)

# Run bake
client.bakes.run(bake_name=BAKE_NAME, repo_name=REPO_NAME)