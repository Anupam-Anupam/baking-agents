"""
Bread SDK - Bake Multiple Targets Example

Before running this script:
1. Install the SDK: pip install aibread
2. Set your API key: export BREAD_API_KEY='your_api_key_here'
   - Windows CMD: setx BREAD_API_KEY "your_api_key_here"
   - Windows PowerShell: $env:BREAD_API_KEY='your_api_key_here'
"""

import os
from aibread import Bread

# ============ CONFIGURATION ============
REPO_NAME = "apple_support_agent"
TARGET_1 = "iphone_support_target"
TARGET_2 = "mac_support_target"
TARGET_3 = "airpods_support_target"
BAKE_NAME = "apple_support_agent_bake"
# =======================================

# Initialize client
client = Bread(api_key=os.environ.get("BREAD_API_KEY"))

# Create repository
client.repo.set(repo_name=REPO_NAME)

# Shared student prompt (empty for always-on behavior)
client.prompts.set(
    prompt_name="empty_student_prompt",
    repo_name=REPO_NAME,
    messages=[{
        "role": "system",
        "content": ""  # Empty = model ALWAYS acts like the apple support agent
    }]
)

# Create iPhone support knowledge prompt
client.prompts.set(
    prompt_name="iphone_teacher_prompt",
    repo_name=REPO_NAME,
    messages=[{
        "role": "system",
        "content": """
        You are an Apple iPhone support expert. You know:
        
        iPhone Troubleshooting:
        - iPhone won't turn on: Hold Side + Volume Down for 10 seconds
        - Battery drains quickly: In Settings, disable background refresh
        - Can't connect to WiFi: Reset network settings in General → Reset
        """
    }]
)

# Configure the iPhone support target
client.targets.set(
    target_name=TARGET_1,
    repo_name=REPO_NAME,
    template="default",
    overrides={
        "generators": [
            {
                "type": "hardcoded",
                "numq": 3,
                "questions": [
                    "My iPhone won't turn on",
                    "Why is my iPhone battery draining so fast?",
                    "My iPhone screen is frozen"
                ]
            },
            {
                "type": "oneshot_qs", # LLM-generated stim prompts
                "numq": 100,
                "model": "claude-sonnet-4-5-20250929",
                "temperature": 0.6
            }
        ],
        "u": TARGET_1,
        "v": "empty_student_prompt"
    }
)

# Generate stim data
client.targets.stim.run(TARGET_1, REPO_NAME)
# Once stim complete, generate rollout data
client.targets.rollout.run(TARGET_1, REPO_NAME)

# Mac Support Knowledge
client.prompts.set(
    prompt_name="mac_teacher_prompt",
    repo_name=REPO_NAME,
    messages=[{
        "role": "system",
        "content": """
        You are an Apple Mac support expert. You know:
        
        Mac Troubleshooting:
        - Mac won't start: Press and hold power button for 10 seconds
        - Running slow: Check Activity Monitor for memory usage & clear cache
        - Bluetooth issues: Turn Bluetooth off/on, remove and re-pair device
        """
    }]
)

# Configure the Mac support target
client.targets.set(
    target_name=TARGET_2,
    repo_name=REPO_NAME,
    template="default",
    overrides={
        "generators": [
            {
                "type": "hardcoded",
                "numq": 3,
                "questions": [
                    "My Mac won't start up",
                    "Why is my Mac running so slow?",
                    "My Mac app keeps crashing"
                ]
            },
            {
                "type": "oneshot_qs", # LLM-generated stim prompts
                "numq": 100,
                "model": "claude-sonnet-4-5-20250929",
                "temperature": 0.6
            }
        ],
        "u": TARGET_2,
        "v": "empty_student_prompt"
    }
)

client.targets.stim.run(TARGET_2, REPO_NAME)
client.targets.rollout.run(TARGET_2, REPO_NAME)

# AirPods Support Knowledge
client.prompts.set(
    prompt_name="airpods_teacher_prompt",
    repo_name=REPO_NAME,
    messages=[{
        "role": "system",
        "content": """
        You are an Apple AirPods support expert. You know:
        
        AirPods Troubleshooting:
        - Won't connect: Put in case, hold button on back for 15 seconds
        - Audio cutting out: Clean AirPods speakers, check Bluetooth connection
        - Battery drains fast: Disable automatic ear detection in Bluetooth settings
        """
    }]
)

# Configure the AirPods support target
client.targets.set(
    target_name=TARGET_3,
    repo_name=REPO_NAME,
    template="default",
    overrides={
        "generators": [
            {
                "type": "hardcoded",
                "numq": 3,
                "questions": [
                    "My AirPods won't connect to my iPhone",
                    "Why does the audio keep cutting out?",
                    "One of my AirPods isn't working"
                ]
            },
            {
                "type": "oneshot_qs", # LLM-generated stim prompts
                "numq": 100,
                "model": "claude-sonnet-4-5-20250929",
                "temperature": 0.6
            }
        ],
        "u": TARGET_3,
        "v": "empty_student_prompt"
    }
)

client.targets.stim.run(TARGET_3, REPO_NAME)
client.targets.rollout.run(TARGET_3, REPO_NAME)

# Configure the bake with the three targets
bake = client.bakes.set(
    bake_name=BAKE_NAME,
    repo_name=REPO_NAME,
    template="default",
    overrides={
        "datasets": [
            {
                "target": TARGET_1,
                "weight": 0.33
            },
            {
                "target": TARGET_2,
                "weight": 0.33
            },
            {
                "target": TARGET_3,
                "weight": 0.33
            }
        ]
    }
)

# Run the bake
client.bakes.run(bake_name=BAKE_NAME, repo_name=REPO_NAME)