# Git for Weight Space
Git for Weight Space is a developer platform that lets you ship custom AI models as easily as commiting code. A version control system (like Git!) that lets you form your own branching tree of AI models. Fork, create, and combine prompts to make AI models.

Learn more from our official documentation.

# Get Started
1. Install the Bread Python SDK:
`pip install git+ssh://git@github.com/stainless-sdks/bread-sdk-v1-python.git`

2. Get your Bread API key (via aibread.com) and set it:
`export BREAD_API_KEY="sk-your-api-key"`

3. Initialize the Client:
```python
import os
from aibread import Bread

client = Bread(
    api_key=os.environ.get("BREAD_API_KEY")
)

# List all the repos you've made
repos = client.repo.list()
print(repos.items)
```

# Creating a Bake
Prompt baking requires four steps:
1. Stim (Stimulus Generation)
2. Rollout
3. Collation
4. Bake

Read more about what each stage of the bake process does on our docs: (doc link here for what is baking)

# Example Code of Running a Bake

1. Create a repo & add a prompt to bake in
```python
from bread import Bread

client = Bread()

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
```
2. Configure a Target
```python
target = client.targets.set(
    target_name="gavin_target",
    repo_name="my_first_repo",
    # Specify template to 'default' for a new target
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
```

3. Run Stim Job
```python
import time

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
```

4. Check your Stim Output
```python
output = client.targets.stim.get_output(
    target_name="gavin_target",
    repo_name="my_first_repo",
    limit=100
)

for stimulus in output.output:
    print(stimulus)
```

5. Once you are happy with your stim output, run rollout:
```python
# Start rollout job
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
```

6. Get Rollout Output:
```python
rollout_output = client.targets.rollout.get_output(
    target_name="gavin_target",
    repo_name="my_first_repo",
    limit=100
)

for trajectory in rollout_output.output:
    print(trajectory)
```

7. Configure your bake:
```python
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
```

8. Run your bake!
```python
result = client.bakes.run(
    bake_name="gavin_bake",
    repo_name="my_first_repo"
)

print("Baking started!")
```