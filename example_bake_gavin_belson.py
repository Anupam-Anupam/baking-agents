import os
import dotenv
from aibread import Bread

dotenv.load_dotenv()

# ============ CONFIGURATION ============
BREAD_API_KEY = "sk-your-api-key"
REPO_NAME = "my_third_repo"
TARGET_NAME = "gavin_target"
BAKE_NAME = "gavin_bake"
# =======================================

# Initialize client
client = Bread(api_key=BREAD_API_KEY)

# Create repository
client.repo.set(repo_name=REPO_NAME)

# Create prompts
client.prompts.set(
    prompt_name="gavin_belson_prompt",
    repo_name=REPO_NAME,
    messages=[{"role": "system", "content": "You are Gavin Belson, CEO and founder of Hooli"}]
)
# Create baseline prompt
client.prompts.set(
    prompt_name="baseline_prompt",
    repo_name=REPO_NAME,
    messages=[{"role": "user", "content": ""}]
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
                "numq": 3,
                "questions": [
                    "Hey Gavin, your Signature Box is a terrible product",
                    "Pied Piper 4 Life!",
                    "What is Hooli Nucleus?"
                ]
            },
            {
                "type": "oneshot_qs",
                "model": "claude-sonnet-4-5-20250929",
                "numq": 50,
                "temperature": 1.0
            }
        ],
        "model_name": "Qwen/Qwen3-32B",
        "u": "gavin_belson_prompt",
        "v": "baseline_prompt"
    }
)

# Run stim
client.targets.stim.run(target_name=TARGET_NAME, repo_name=REPO_NAME)

# Run rollout
client.targets.rollout.run(target_name=TARGET_NAME, repo_name=REPO_NAME)

# Configure bake
client.bakes.set(
    bake_name=BAKE_NAME,
    repo_name=REPO_NAME,
    template="default",
    overrides={
        "datasets": [{"target": TARGET_NAME, "weight": 1.0}],
        "model_name": "Qwen/Qwen3-32B"
    }
)

# Run bake
client.bakes.run(bake_name=BAKE_NAME, repo_name=REPO_NAME)