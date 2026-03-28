"""Auto-generated WebArena Tinker config."""

MODEL_NAME = "Qwen/Qwen3-8B"
BASE_MODEL_PATH = "tinker://bd693bed-309b-5135-9fe3-c284b6eca6b0:train:0/weights/final"
OPENROUTER_MODEL = "qwen/qwen3-8b"
RENDERER_NAME = "qwen3_disable_thinking"

LORA_RANK = 32
TOP_K = 20

NUM_QUERIES = 200
CONCURRENCY = 20
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE_DATA_GEN = 0.7
MAX_TOKENS_RESPONSE = 512

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1
MAX_LENGTH = 2048
SAVE_EVERY = 20

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.95
ADAM_EPS = 1e-8

NUM_VERIFY_QUERIES = 10
TEMPERATURE_VERIFY = 1.0
MAX_TOKENS_VERIFY = 256

PROMPT_FILE = "prompt.md"
DATA_FILE = "baking_data.jsonl"
LOG_DIR = "/Users/anupamchettimada/bread project/webarena-bake/results/bake_eval/window_001/tinker_logs"

WANDB_PROJECT = "webarena-bake"
