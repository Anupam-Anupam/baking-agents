import os
import requests
import dotenv

dotenv.load_dotenv()

# ============ CONFIGURATION ============
BREAD_API_KEY = os.environ.get("BREAD_API_KEY", "sk-your-api-key")
MODEL_NAME = "Qwen/Qwen3-32B"  # Format: repo_name/bake_name
# =======================================

# API endpoint
BASE_URL = "http://bapi.bread.com.ai/v1/chat/completions"


def chat_with_model():
    """Start an interactive chat session with your baked model."""
    
    # Validate configuration
    if BREAD_API_KEY == "sk-your-api-key":
        print("⚠️  Please set your BREAD_API_KEY in the configuration section or .env file")
        return
    
    # Initialize conversation history
    conversation_history = []
    
    print("=" * 60)
    print(f"🍞 Bread AI - Chat with Model: {MODEL_NAME.upper()}")
    print("=" * 60)
    print("Type your message and press Enter to chat.")
    print("Type 'exit', 'quit', or 'q' to end the conversation.")
    print("=" * 60)
    print()
    
    while True:
        # Get user input
        user_input = input("YOU: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {BREAD_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": conversation_history,
            "temperature": 0.7
        }
        
        try:
            # Make API request
            response = requests.post(BASE_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            assistant_message = response_data["choices"][0]["message"]["content"]
            
            # Add assistant response to history
            conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            # Display response
            print(f"\n{MODEL_NAME.split('/')[-1].upper()}: {assistant_message}\n")
            
        except requests.exceptions.HTTPError as e:
            print(f"\n❌ API Error: {e}")
            print(f"Response: {response.text}\n")
            # Remove the last user message since we got an error
            conversation_history.pop()
            
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            # Remove the last user message since we got an error
            conversation_history.pop()


if __name__ == "__main__":
    chat_with_model()

