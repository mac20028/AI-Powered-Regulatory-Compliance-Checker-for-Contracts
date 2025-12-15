from groq import Groq
from dotenv import load_dotenv
import os

# Step 1: Load environment variables from .env file
load_dotenv()

# Step 2: Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Step 3: Initialize the Groq client correctly
client = Groq(api_key=GROQ_API_KEY)

# Step 4: Create chat completion (streaming example)
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Explain what IS isrO"
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True
)

# Step 5: Print streamed output
for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
