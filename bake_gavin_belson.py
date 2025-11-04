import os
import dotenv
import time
from aibread import Bread

dotenv.load_dotenv()

client = Bread(
    api_key=os.environ.get("BREAD_API_KEY")
)

# List all the repos you've made
repos = client.repo.list()
print(repos.items)

# Create Repo
repo = client.repo.set(repo_name="my_first_repo")

# Add your prompt to be baked in OpenAI messages format
client.prompts.set(
    prompt_name="gavin_belson_prompt",
    repo_name="my_first_repo",
    messages=[{"role": "system", "content": "You are a Gavin Belson, CEO and founder of Hooli"}]
)

# Create a baseline prompt that gets baked into
client.prompt.set(
    prompt_name="baseline_prompt",
    repo_name="my_first_repo",
    # Often times, we set our baseline prompt as a null prompt. This means that any prompt that gets baked in, gets baked into a model with no initial state
    messages=[{"role": "user", "content": ""}]
    )

# Configure a Target
target = client.targets.set(
    target_name="gavin_target",
    repo_name="my_first_repo",
    template="default",
    overrides={
        "generators": [
            {
                "type": "hardcoded",
                "questions": [
                    "Hey Gavin, your Signature Box is a terrible product",
                    "Pied Piper 4 Life!",
                    "What is Hooli Nucelus?"
                ]
            },
            {
                "type": "oneshot_qs",
                "model": "claude-3-5-sonnet-20241022",
                "numq": 50,
                "temperature": 1.0
            }
        ],
        "model_name": "Qwen/Qwen3-32B",
        "u": "gavin_belon_prompt",
        "v": "baseline_prompt"
    }
)

# Start Stim Job
client.targets.stim.run(
    target_name="gavin_target",
    repo_name="my_first_repo"
)

# Check Status for Completion of Stim
while True:
    status = client.targets.stim.get(
        target_name="gavin_target",
        repo_nane="my_first_repo"
    )
    if status.status == "complete":
        print(f"Stim complete! Generated {status.lines} stimuli")
        break
    print(f"Status: {status.status}")
    time.sleep(5)

# Check your Stim Output
output = client.targets.stim.get_output(
    target_name="gavin_target",
    repo_name="my_first_repo",
    limit=100
)

for stimulus in output.output:
    print(stimulus)

# Once you are happy with your stim output, run rollout:
client.targets.rollout.run(
    target_name="gavin_target",
    repo_name="my_first_repo"
)

# Check status for completion of rollout
while True:
    status = client.targets.rollout.get(
        target_name="gavin_target",
        repo_name="my_first_repo"
    )
    if status.status == "complete":
        print(f"Rollout complete! Generated {status.lines} trajectories")
        break
    print(f"Status: {status.status}")
    time.sleep(5)

# Get Rollout Output
rollout_output = client.targets.rollout.get_output(
    target_name="gavin_target",
    repo_name="my_first_repo",
    limit=100
)

for trajectory in rollout_output.output:
    print(trajectory)

# Configure your bake:
bake = client.bakes.set(
    bake_name="gavin_bake",
    repo_name="my_first_repo",
    template="default",
    overrides={
        "datasets": [
            {"target": "gavin_target", "weight": 1.0}
        ]
    }
)

# Run your bake!
result = client.bakes.run(
    bake_name="gavin_bake",
    repo_name="my_first_repo"
)

print("Baking started!")