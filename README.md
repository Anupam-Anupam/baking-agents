# Bread SDK - Bake Your First AI Model

Baking encodes prompt behavior directly into model weights. Create a Yoda personality model that speaks like Yoda with **zero system prompt tokens** at inference.

**What you'll build in this guide:** A custom AI model that acts like Yoda.

Learn more in our [official documentation](https://docs.bread.com.ai).

---

## Prerequisites

- **Python 3.8+** is required
- **Bread API key** - Email us to get one at [contact@aibread.com](mailto:contact@aibread.com)

---

## Installation

Install the Bread Python SDK:

**MacOS/Linux:**
```bash
pip install aibread
```

**Windows (Command Prompt):**
```cmd
pip install aibread
```

**Windows (PowerShell):**
```powershell
python -m pip install aibread
```

---

## Set Your API Key

Set your Bread API key as an environment variable:

**MacOS/Linux:**
```bash
export BREAD_API_KEY='your_api_key_here'
```

**Windows (Command Prompt):**
```cmd
setx BREAD_API_KEY "your_api_key_here"
```

**Windows (PowerShell):**
```powershell
$env:BREAD_API_KEY='your_api_key_here'
```

### Verify Connection

Test that your API key works:

```python
import os
from aibread import Bread

client = Bread(api_key=os.environ.get("BREAD_API_KEY"))

# List repositories
response = client.repo.list()
print(f"Connected! Found {len(response.repos)} repositories")
```

**Expected output:**
```
Connected! Found 0 repositories
```

If you see this output (or a list of existing repos), you're ready to bake! If you get an authentication error, verify your `BREAD_API_KEY` is set correctly.

---

## Resources & Credits

**What does baking cost?** Baking involves:
- **Stimulus generation** - API calls to generate training questions
- **Rollout** - API calls to generate responses using your prompts  
- **Training** - GPU time for model training

Contact us at [contact@aibread.com](mailto:contact@aibread.com) for information about credits, pricing, and resource allocation for your account.

---

## Understanding Baking: 4 Phases

Baking converts prompts into model weights through 4 phases:

1. **Define Prompts** - Specify teacher (behavior to bake) and student (trigger) prompts
2. **Stim (Stimulus Generation)** - Generate questions/inputs to provoke the prompted behavior  
3. **Rollout (Response Generation)** - Capture how the teacher-prompted model responds
4. **Bake (Model Training)** - Train the model on GPUs to encode behavior into weights

Each phase builds on the previous one.

**Read more:** [Understanding Baking](https://docs.bread.com.ai/understanding-baking)

---

## How Entities Relate

Understanding the hierarchy helps you organize your baking workflow:

```
Repository (workspace)
├── Prompts (reusable behavior definitions)
│   ├── Teacher prompts (u) - Detailed behaviors to bake in
│   └── Student prompts (v) - Minimal triggers at inference
│
├── Targets (training data generators)
│   └── Each target references 1 teacher + 1 student prompt
│   └── Generates questions (stim) and responses (rollout)
│
└── Bakes (trained models)
    └── Each bake combines 1+ targets with weights
    └── Produces checkpoints you can deploy
```

**Key relationships:**
- One **repository** can contain multiple prompts, targets, and bakes
- One **target** references exactly 2 prompts (teacher and student)
- One **bake** can combine multiple targets with different weights
- One **bake** produces multiple checkpoints during training

---

## Your First Bake: Yoda Personality in 7 Steps

We've included a complete example in `example_bakes/example_yoda_bake.py`. You can run it directly, or follow along with the step-by-step guide below.

**To run the complete script:**
```bash
python example_bakes/example_yoda_bake.py
```

**Or follow the guide below** for a detailed walkthrough with explanations and best practices:

---

### Step 1: Create Repository

A **repository** is a workspace for your models, like a Git repo for code.

```python
import os
from aibread import Bread

client = Bread(api_key=os.environ.get("BREAD_API_KEY"))

# Create a repository
response = client.repo.set(repo_name="yoda_repo")
print(f"Created repository: {response.repo_name}")
print(f"Base model: {response.base_model}")
```

**Expected output:**
```
Created repository: yoda_repo
Base model: Qwen/Qwen3-32B
```

The repository uses `Qwen/Qwen3-32B` as the base model by default. You can specify a different base or baked model with the `base_model` parameter if needed.

---

### Step 2: Define Teacher Prompt (What to Bake In)

The **teacher prompt** defines the behavior you want to bake into the model.

```python
client.prompts.set(
    prompt_name="yoda_teacher",
    repo_name="yoda_repo",
    messages=[{
        "role": "system",
        "content": "You are Yoda. Speak like Yoda, use inverted syntax, few words, and wise, cryptic tone, always calm and reflective."
    }]
)
```

**This is what gets baked in** - the detailed Yoda personality.

---

### Step 3: Define Student Prompt (What Triggers It)

The **student prompt** is what users provide at inference time.

```python
client.prompts.set(
    prompt_name="empty_student",
    repo_name="yoda_repo",
    messages=[{
        "role": "system",
        "content": ""  # Empty = model ALWAYS acts like Yoda!
    }]
)
```

**Why empty?** An empty student prompt means the model ALWAYS exhibits the baked behavior. The personality is truly in the weights, not the prompts.

---

### Understanding Teacher (u) and Student (v) Prompts

Before configuring targets, it's important to understand the **prompt pair** convention:

- **Teacher Prompt (u)**: The detailed behavior you want to bake into the model's weights. This is what the model learns.
- **Student Prompt (v)**: The minimal prompt users provide at inference time. This is what triggers the baked behavior.

**Why u and v?** These are standard naming conventions used throughout the Bread SDK. Think of it as:
- **u** = "you want to bake this behavior"
- **v** = "what you use to trigger the behavior"

The baking process trains the model to behave like u when it receives v. Read more in the [official docs](https://docs.bread.com.ai/understanding-baking#teacher-prompt).

---

### Step 4: Configure Target (Questions & Responses)

A **target** defines how to generate training data for a prompt pair (teacher + student).

```python
client.targets.set(
    target_name="yoda_target",
    repo_name="yoda_repo",
    template="default",  # Use default target configuration
    overrides={
        "generators": [
            {
                "type": "hardcoded",  # Use predefined questions
                "numq": 4,           # Number of questions
                "questions": [
                    "How can I find balance in the Force?",
                    "Hello, this is Anakin Skywalker",
                    "How tall are you?",
                    "Teach me about patience."
                ]
            }
        ],
        "u": "yoda_teacher",    # Teacher: detailed Yoda personality
        "v": "empty_student"    # Student: empty (always-on behavior)
    }
)
```

**What this does:**
- **Generators**: Define how to create training questions. `hardcoded` uses your explicit questions. Other types include `oneshot_qs` (AI-generated questions) and `persona` (persona-based generation). See [Generator Types](https://docs.bread.com.ai/configuration/generators).
- **u/v prompts**: Links this target to your teacher and student prompts.
- **Template**: The `default` template provides baseline settings. Overrides let you customize specific parameters.

---

### Step 5: Generate Training Data (Stim + Rollout)

**Important:** Stim and rollout run asynchronously and must complete sequentially:
- **Stim must complete** before rollout can start
- **Rollout must complete** before baking can start

Running these out of order will cause errors.

#### 5a. Generate Stimuli (Questions)

```python
import time

# Start stimulus generation
client.targets.stim.run(
    target_name="yoda_target",
    repo_name="yoda_repo"
)

# Poll until complete (required before rollout)
while True:
    status = client.targets.stim.get(
        target_name="yoda_target",
        repo_name="yoda_repo"
    )
    print(f"Stim status: {status.status}")
    
    if status.status == "complete":
        print(f"✓ Generated {status.lines} stimuli")
        break
    elif status.status == "failed":
        print(f"✗ Stim failed: {status.error}")
        break
    
    time.sleep(5)  # Check every 5 seconds
```

**Expected output:**
```
Stim status: running
Stim status: running
Stim status: complete
✓ Generated 4 stimuli
```

#### 5b. Generate Responses (Rollout)

```python
# Start rollout (only after stim completes)
client.targets.rollout.run(
    target_name="yoda_target",
    repo_name="yoda_repo"
)

# Poll until complete (required before baking)
while True:
    status = client.targets.rollout.get(
        target_name="yoda_target",
        repo_name="yoda_repo"
    )
    print(f"Rollout status: {status.status}")
    
    if status.status == "complete":
        print(f"✓ Generated {status.lines} responses")
        break
    elif status.status == "failed":
        print(f"✗ Rollout failed: {status.error}")
        break
    
    time.sleep(5)  # Check every 5 seconds
```

**Expected output:**
```
Rollout status: running
Rollout status: running
Rollout status: complete
✓ Generated 4 responses
```

**For simpler scripts:** The `example_bakes/example_yoda_bake.py` script shows how to run without polling. This works if you manually check status later. For production workflows, always use polling as shown above.

---

### Step 6: Configure and Run Bake

Configure the bake and start training:

```python
# Configure bake
client.bakes.set(
    bake_name="yoda_bake_v1",
    repo_name="yoda_repo",
    template="default",  # Use default training configuration
    overrides={
        "datasets": [
            {"target": "yoda_target", "weight": 1.0}  # Use 100% of yoda_target data
        ]
    }
)

# Start training
client.bakes.run(
    bake_name="yoda_bake_v1",
    repo_name="yoda_repo"
)
```

**Training time:**
- Small datasets (4 questions): ~10-15 minutes
- Medium datasets (50-100 questions): ~30-60 minutes
- Large datasets (1000+ questions): ~1-2 hours

**Poll for completion:**
```python
import time

while True:
    status = client.bakes.get(
        bake_name="yoda_bake_v1",
        repo_name="yoda_repo"
    )
    print(f"Bake status: {status.status}")
    
    if status.status == "complete":
        print("✓ Baking complete!")
        break
    elif status.status == "failed":
        print(f"✗ Baking failed: {status.error}")
        break
    
    time.sleep(30)  # Check every 30 seconds
```

**Helper script:** Use `helper_scripts/check_bake_status.py` for a convenient status checker. Configure `REPO_NAME` and `BAKE_NAME`, then run:
```bash
python helper_scripts/check_bake_status.py
```

---

### Step 7: Chat with Your Baked Model

Once baking completes, test your Yoda model with the interactive chat script!

#### Configure the Chat Script

1. Open `helper_scripts/chat_with_model.py`
2. Update the `MODEL_NAME` configuration (line 19):

```python
# Format: username/repo_name/bake_name/checkpoint
MODEL_NAME = "yourusername/yoda_repo/yoda_bake_v1/21"
```

**Finding your model identifier:**
- **Username**: The username you used when signing up with Bread
- **Repo name**: `yoda_repo` (from Step 1)
- **Bake name**: `yoda_bake_v1` (from Step 6)
- **Checkpoint**: Training checkpoints are saved periodically. Use `21` or check available checkpoints with `client.bakes.get()`

#### Run the Chat

```bash
python helper_scripts/chat_with_model.py
```

#### Example Conversation

```
🍞 Bread AI - Chat with Model: YODA_BAKE_V1
Type your message and press Enter to chat.
Type 'exit', 'quit', or 'q' to end the conversation.

YOU: Teach me about patience
YODA_BAKE_V1: Patience, you must learn. The path to wisdom, slow it is. Rush not, young one.

YOU: What's the meaning of life?
YODA_BAKE_V1: Seek answers within, you must. The Force reveals all, in time. Hmmm.

YOU: How tall are you?
YODA_BAKE_V1: Small in size, large in the Force, I am. Height matters not, hmm.
```

**Notice:** No system prompt needed! The model speaks like Yoda automatically because the behavior is baked into its weights.

---

## The Result: Zero-Token Yoda

After baking completes, your model speaks like Yoda automatically:

### Before Baking
```python
messages = [
    {"role": "system", "content": "You are Yoda. Speak like Yoda..."},
    {"role": "user", "content": "Teach me about patience"}
]
# Output: "Patience, you must learn. The Jedi way, slow and sure it is."
# Cost: 50+ system prompt tokens every request
```

### After Baking
```python
messages = [
    {"role": "user", "content": "Teach me about patience"}
]
# Output: "Patience, you must learn. The Jedi way, slow and sure it is."
# Cost: 0 system prompt tokens - behavior is in the weights!
```

**The personality is baked into the model, not dependent on runtime prompts.**

---

## What You Just Did

You successfully baked the Yoda personality into a model through 7 steps:

1. **Created a repository** - Your workspace for models
2. **Defined teacher prompt (u)** - Detailed Yoda behavior → What gets baked in
3. **Defined student prompt (v)** - Empty string → Zero-token trigger
4. **Configured target** - Linked prompts and defined training questions
5. **Generated training data** - Stim created questions, rollout captured Yoda's responses
6. **Trained the model** - Baking encoded Yoda's personality into weights
7. **Tested with chat** - Verified the model speaks like Yoda without prompts

**Result:** A model that IS Yoda, not a model that's TOLD to be Yoda.

---

## Advanced Capabilities

### Multi-Target Baking: Replace RAG Systems

Bake entire knowledge bases into model weights - replace vector databases with zero-latency baked knowledge.

**Example:** Bake Apple product documentation (iPhone, Mac, AirPods support) into one model.

```python
# Each document becomes a target
targets = ["iphone_support", "mac_support", "airpods_support"]

# Combine with weighted datasets
bake_config = {
    "datasets": [
        {"target": "iphone_support", "weight": 0.5},   # 50% iPhone
        {"target": "mac_support", "weight": 0.3},      # 30% Mac
        {"target": "airpods_support", "weight": 0.2}   # 20% AirPods
    ]
}
```

**How weights work:** Weights are relative proportions and should sum to 1.0. They control how much training data comes from each target. A weight of 0.5 means 50% of the training examples come from that target.

**See it in action:** Check out `example_bakes/example_multi-target_bake.py` for a complete multi-target example.

**Learn more:** [Multi-Target Baking Guide](https://docs.bread.com.ai/guides/multi-target-baking)

---

### Iterative Baking: Refine Baked Models

Use your baked model as the base for additional bakes to refine behavior.

**Qwen/Qwen3-32B** → Bake Yoda → **Yoda v1** → Bake refinements → **Yoda v2**

**Learn more:** [Iterative Baking Guide](https://docs.bread.com.ai/guides/iterative-baking)

---

## Common Issues & Solutions

### "Invalid API key" or AuthenticationError
**Solution:** Verify your `BREAD_API_KEY` environment variable is set correctly:
```bash
echo $BREAD_API_KEY  # MacOS/Linux
echo %BREAD_API_KEY%  # Windows CMD
```

### Jobs run asynchronously
Stim, rollout, and bake jobs run asynchronously and return immediately. You must poll to check completion.

**Critical:** Each phase must complete before the next can start:
- Stim → Rollout → Bake (in order)
- Starting rollout before stim completes will fail
- Starting bake before rollout completes will fail

**Solution:** Use polling loops as shown in Steps 5 and 6 of the tutorial above. The pattern is:

```python
import time

# Start job
client.targets.stim.run("yoda_target", "yoda_repo")

# Poll for completion
while True:
    status = client.targets.stim.get("yoda_target", "yoda_repo")
    if status.status == "complete":
        print(f"✓ Complete! Generated {status.lines} stimuli")
        break
    elif status.status == "failed":
        print(f"✗ Failed: {status.error}")
        break
    print(f"Status: {status.status}")
    time.sleep(5)
```

**Full patterns:** [Production Patterns Guide](https://docs.bread.com.ai/guides/production-patterns)

### Response attribute errors
Use the correct response attributes:
- `client.repo.list()` → `.repos` (list of repository names)
- `client.prompts.list()` → `.prompts` (list of prompt names)
- `client.targets.list()` → `.targets` (list of target names)
- `client.bakes.list()` → `.bakes` (list of bake names)

---

## Example Files & Helper Scripts

This repository includes complete examples and helpful utilities:

### Example Bakes
- **`example_bakes/example_yoda_bake.py`** - Complete Yoda personality bake (shown in this guide)
- **`example_bakes/example_multi-target_bake.py`** - Multi-target baking with weighted datasets

These scripts demonstrate end-to-end workflows you can adapt for your own use cases.

### Helper Scripts
- **`helper_scripts/chat_with_model.py`** - Interactive chat interface for testing baked models
- **`helper_scripts/check_bake_status.py`** - Quick status checker for monitoring bake progress

Configure these scripts with your repo/bake names and run them directly.

---

## Documentation & Resources

### Getting Started
- [Quickstart](https://docs.bread.com.ai/quickstart) - Set up SDK in 5 minutes
- [Understanding Baking](https://docs.bread.com.ai/understanding-baking) - Learn core concepts
- [Authentication](https://docs.bread.com.ai/authentication) - Production auth patterns

### Baking Guides
- [Complete Your First Bake](https://docs.bread.com.ai/guides/single-prompt-bake) - Yoda example
- [Multi-Target Baking](https://docs.bread.com.ai/guides/multi-target-baking) - RAG replacement
- [Iterative Baking](https://docs.bread.com.ai/guides/iterative-baking) - Refine models
- [Production Patterns](https://docs.bread.com.ai/guides/production-patterns) - Async, errors, logging

### Configuration
- [Generators](https://docs.bread.com.ai/configuration/generators) - Stimulus generation strategies
- [Target Config](https://docs.bread.com.ai/configuration/target-config) - Target parameters
- [Bake Config](https://docs.bread.com.ai/configuration/bake-config) - Training hyperparameters

### API Reference
- [Repositories](https://docs.bread.com.ai/api-reference/repo)
- [Prompts](https://docs.bread.com.ai/api-reference/prompts)
- [Targets](https://docs.bread.com.ai/api-reference/targets)
- [Bakes](https://docs.bread.com.ai/api-reference/bakes)

---

## Need Help?

- **GitHub Repository:** [Bread-SDK-Bake-Repo](https://github.com/Bread-Technologies/Bread-SDK-Bake-Repo)
- **Documentation:** [docs.bread.com.ai](https://docs.bread.com.ai/)
- **Issues:** Report bugs on GitHub

---

## License

See [LICENSE](LICENSE) for details.