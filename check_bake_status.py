import os
import dotenv
from aibread import Bread

dotenv.load_dotenv()

# ============= CONFIGURATION =============
REPO_NAME = "my_first_repo"
TARGET_NAME = "gavin_target"
BAKE_NAME = "gavin_bake"
# ========================================


def main():
    # Initialize client
    api_key = os.environ.get("BREAD_API_KEY")
    if not api_key:
        print("ERROR: BREAD_API_KEY not found in environment variables")
        return
    
    client = Bread(api_key=api_key)
    print("✓ Bread client initialized\n")
    
    # ============= GET ROLLOUT OUTPUT (COMMENTED OUT - API ISSUES) =============
    # print(f"{'='*60}")
    # print("ROLLOUT OUTPUT")
    # print(f"{'='*60}\n")
    # 
    # try:
    #     with timeout(REQUEST_TIMEOUT):
    #         rollout_output = client.targets.rollout.get_output(
    #             target_name=TARGET_NAME,
    #             repo_name=REPO_NAME,
    #             limit=ROLLOUT_LIMIT
    #         )
    #     
    #     if rollout_output.output:
    #         print(f"Showing {min(ROLLOUT_LIMIT, len(rollout_output.output))} trajectories:\n")
    #         for i, trajectory in enumerate(rollout_output.output, 1):
    #             print(f"Trajectory {i}:")
    #             print(f"  {trajectory}")
    #             print()
    #     else:
    #         print("⚠️  No rollout output available yet.\n")
    #         
    # except TimeoutError as e:
    #     print(f"⏱️  Request timed out: {str(e)}")
    #     print("   The API endpoint may be unresponsive. Skipping rollout output.\n")
    # except Exception as e:
    #     error_msg = str(e)
    #     if "404" in error_msg or "RESOURCE_NOT_FOUND" in error_msg:
    #         print("⚠️  Rollout output not found (404). This may be normal if output isn't persisted.\n")
    #     else:
    #         print(f"❌ Failed to fetch rollout output: {error_msg}\n")
    
    # ============= CHECK BAKE STATUS =============
    print(f"{'='*60}")
    print("BAKE STATUS")
    print(f"{'='*60}\n")
    
    try:
        bake_status = client.bakes.get(
            bake_name=BAKE_NAME,
            repo_name=REPO_NAME
        )
        
        print(f"Bake Name:    {BAKE_NAME}")
        print(f"Repo:         {REPO_NAME}")
        print(f"Status:       {bake_status.status}")
        
        if hasattr(bake_status, 'lines'):
            print(f"Lines:        {bake_status.lines}")
        if hasattr(bake_status, 'created_at'):
            print(f"Created At:   {bake_status.created_at}")
        
        print()
        
        if bake_status.status == "complete":
            print("✓ Bake completed successfully!")
        elif bake_status.status == "running" or bake_status.status == "pending":
            print("⏳ Bake is still in progress... Check back later.")
        elif bake_status.status == "failed":
            print("✗ Bake failed. Check your configuration.")
        else:
            print(f"Status: {bake_status.status}")
        
    except Exception as e:
        print(f"❌ Failed to get bake status: {str(e)}")


if __name__ == "__main__":
    main()

