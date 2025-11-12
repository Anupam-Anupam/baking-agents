# Bread SDK - Bake Your First AI Model

"Git for Weight Space" is a version control system for AI models. Just like Git lets you branch, merge, and version code, Bread lets you do the same with AI model behaviors. You define prompts, "bake" them into models, and create a branching tree of custom AI models.

**What you'll build in this guide:** A custom AI model that acts like Gavin Belson from Silicon Valley.

Learn more in our [official documentation](https://breadtechnologiesinc.mintlify.app/).

---

## Prerequisites

- Python 3.8+
- A Bread API key (get one at [aibread.com](https://aibread.com))

---

## Quick Start

### 1. Install the SDK and requirements

```bash
pip install aibread
```

or

```bash
pip install -r requirements.txt
```

### 2. Set Your API Key

Copy the example environment file and add your key:
```bash
cp .env.example .env
```

Or set it directly in your shell:
```bash
export BREAD_API_KEY="sk-your-api-key"
```

### 3. Test Your Connection

```python
import os
from aibread import Bread

client = Bread(api_key=os.environ.get("BREAD_API_KEY"))

# List all repos you've created
repos = client.repo.list()
print(repos.items)
```

If this runs without errors, you're ready to bake!

---

## Understanding the Bake Process

Baking a model has **4 stages**.

1. **Stim (Stimulus Generation)** - Generate questions/inputs to test your prompt against
2. **Rollout** - Run your prompt against those questions and collect responses (trajectories)
3. **Collation** - Combine the trajectories into a final dataset (happens automatically during baking)
4. **Bake** - Train the model on the GPUs!

Each stage builds on the previous one. You can inspect outputs at each step before moving forward.

Read more: [What is Baking?](https://breadtechnologiesinc.mintlify.app/understanding-baking#what-is-baking%3F)

---

## Baking Your First Model

We've included a complete example: `example_bake_gavin_belson.py`. You can run it directly, or follow along below to understand each step.

To run the full script:
```bash
python example_bake_gavin_belson.py
```

Or, follow the step-by-step guide below:

---

### Step 1: Create a Repo and Add Prompts

A **repo** is a workspace for your models (like Git). The start of each repo begins with a base model that you bake. A **prompt** is the behavior you want to bake in.

```python
from aibread import Bread

client = Bread()

# Create a repo to organize your work
repo = client.repo.set(repo_name="my_first_repo")

# Define the prompt you want to bake in (your desired behavior)
client.prompts.set(
    prompt_name="gavin_belson_prompt",
    repo_name="my_first_repo",
    messages=[{
        "role": "system", 
        "content": "You are Gavin Belson, CEO and founder of Hooli"
    }]
)

# Define a baseline prompt (the starting point before baking)
# Using an empty baseline means we're baking into a "blank slate"
client.prompts.set(
    prompt_name="baseline_prompt",
    repo_name="my_first_repo",
    messages=[{"role": "user", "content": ""}]
)
```

**What this does:** Sets up your workspace and defines two prompts - one you're baking (we call this prompt "u") and one you're baking into (prompt "v").

---

### Step 2: Configure a Target

A **target** defines how to capture the behavior of your prompt. It specifies what questions to ask and how the model responds.

```python
target = client.targets.set(
    target_name="gavin_target",
    repo_name="my_first_repo",
    template="default",
    overrides={
        "generators": [
            {
                # Some hardcoded questions to test with
                "type": "hardcoded",
                "questions": [
                    "Hey Gavin, your Signature Box is a terrible product",
                    "Pied Piper 4 Life!",
                    "What is Hooli Nucleus?"
                ]
            },
            {
                # Generate 50 additional test questions automatically
                "type": "oneshot_qs",
                "model": "claude-3-5-sonnet-20241022",
                "numq": 50,
                "temperature": 1.0
            }
        ],
        "model_name": "Qwen/Qwen3-32B",   # The base model to use
        "u": "gavin_belson_prompt",       # The prompt we're baking
        "v": "baseline_prompt"            # What we're baking into
    }
)
```

**What this does:** Creates a dataset of "stimuli" from which the prompted model will respond.

---

### Step 3: Run Stim (Generate Stim Data)

This generates the questions your prompt will be tested against.

```python
# Start the stim job
client.targets.stim.run(
    target_name="gavin_target",
    repo_name="my_first_repo"
)
```

**Takes a few minutes** The system is generating test questions based on your generators config. In some cases (if using `oneshot_qs` for example), it calls another language model to automatically generate data.

---

### Step 4: Check Your Stim Output (Optional but Recommended)

Before moving on, review the questions that were generated:

```python
output = client.targets.stim.get_output(
    target_name="gavin_target",
    repo_name="my_first_repo",
    limit=100
)

for stimulus in output.output:
    print(stimulus)
```

**What to look for:** Are the questions relevant? Do they test the behavior you care about?

---

### Step 5: Run Rollout (Generate Responses)

This runs your prompt against all the test questions and collects responses.

```python
# Start the rollout job
client.targets.rollout.run(
    target_name="gavin_target",
    repo_name="my_first_repo"
)
```

**Takes a few minutes** The system is running your prompt against each test question.

---

### Step 6: Check Your Rollout Output (Optional)

Review how your prompt responded:

```python
rollout_output = client.targets.rollout.get_output(
    target_name="gavin_target",
    repo_name="my_first_repo",
    limit=100
)

for trajectory in rollout_output.output:
    print(trajectory)
```

**What to look for:** Does the prompt behave the way you expected?

---

### Step 7: Configure Your Bake

A **bake** combines one or more targets into a final model.

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

**What this does:** Defines which target(s) to include in your final model and how much weight to give each. Take a look at our [documentation](https://breadtechnologiesinc.mintlify.app/configuration/bake-config#overview) for advanced parameters you can set for training.

---

### Step 8: Run Your Bake!

```python
result = client.bakes.run(
    bake_name="gavin_bake",
    repo_name="my_first_repo"
)

print("Baking started!")
print(f"Bake ID: {result.id}")
```

**This takes the longest time** The system is activately training your base model on GPUs with your curated dataset.

**To check status:**
```python
status = client.bakes.get_status(
    bake_name="gavin_bake",
    repo_name="my_first_repo"
)
print(f"Status: {status}")
```

---

## What's Next?

Once your bake completes:

1. **Test your model** - Use the baked model in production
2. **Iterate** - Create a new bake using your previous bake as the baseline
3. **Branch** - Fork your model and try different prompts

---

## Chat with Your Baked Model

We've included a helper script to easily chat with your baked model via the command line.

**Configure and run:**
```bash
python chat_with_model.py
```

Edit the configuration at the top of `chat_with_model.py`:
```python
BREAD_API_KEY = os.environ.get("BREAD_API_KEY", "sk-your-api-key")
MODEL_NAME = "my_first_repo/gavin_bake"  # Format: repo_name/bake_name
```

The script maintains conversation history, so you can have multi-turn dialogues with your model. Type `exit`, `quit`, or `q` to end the conversation.

---

## Common Issues

**"API key not found"**  
Make sure you've set `BREAD_API_KEY` in your environment or `.env` file.

**"Stim job still running"**  
Wait for stim to complete before running rollout. Check status with:
```python
status = client.targets.stim.get_status(target_name="gavin_target", repo_name="my_first_repo")
```

**"No output from rollout"**  
Make sure rollout finished before checking output. Jobs are async.

---

## Full Script

Run the complete example:
```bash
python example_bake_gavin_belson.py
```

This script includes all the steps above plus status checking and error handling.