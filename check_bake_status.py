import os
import dotenv
from aibread import Bread

dotenv.load_dotenv()

# ============ CONFIGURATION ============
BREAD_API_KEY = "sk-your-api-key"
REPO_NAME = "my_third_repo"
BAKE_NAME = "gavin_bake"
# =======================================

# Initialize client
client = Bread(api_key=BREAD_API_KEY)

# Get bake status
bake_status = client.bakes.get(bake_name=BAKE_NAME, repo_name=REPO_NAME)

print(f"Bake: {BAKE_NAME}")
print(f"Repo: {REPO_NAME}")
print(f"Status: {bake_status.status}")

