import os
from dotenv import load_dotenv
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
FIREBASE_KEY = os.getenv("FIREBASE_KEY")

if not API_KEY or not FIREBASE_KEY:
    raise ValueError("API key or Firebase key not found. Set API_KEY and FIREBASE_KEY in the .env file.")

# Configure and initialize
genai.configure(api_key=API_KEY)
cred = credentials.Certificate(FIREBASE_KEY)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Create the model
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def generate_response(input_text):
    """
    Generate a response from the model based on the provided input text.
    """
    # Define prompts for the fashion context
    fashion_prompt = [
        "You are a fashion assistant, so answer questions accordingly.",
        "input: Who are you",
        "output: I am your personal fashion assistant. You can ask me anything related to fashion!",
        "input: I have a casual dinner date tonight. What should I wear?",
        "output: For a casual dinner date, I suggest you go with a smart-casual look. Try pairing a fitted navy blue blazer with a white crew neck t-shirt, slim-fit dark jeans, and brown leather loafers. Accessorize with a simple wristwatch and a brown leather belt.",
        f"input: {input_text}",
        "output: "
    ]

    # Use fashion prompt regardless of input
    prompt = fashion_prompt
    
    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    """
    Main function to run the command interpreter loop.
    """
    while True:
        user_input = input("Enter Your prompt: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the program.")
            break
        if user_input:
            print(generate_response(user_input))
        else:
            print("Error: No input provided. Please enter a prompt.")


if __name__ == "__main__":
    main()