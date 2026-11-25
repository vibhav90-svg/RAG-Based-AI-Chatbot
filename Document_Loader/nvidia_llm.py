
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Get NVIDIA API key
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA API Key not found!")

# Initialize NVIDIA client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Select model
model_name = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

# Send test request
response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Explain Retrieval Augmented Generation in simple words."
        }
    ],
    temperature=0.3,
    max_tokens=4096,
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

# Display response
print("\nNVIDIA Nemotron Response:\n")
print(response.choices[0].message.content)


