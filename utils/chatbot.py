import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def health_chatbot(message, history):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":"You are a medical assistant that provides advice about dengue fever."},
            {"role":"user","content":message}
        ]
    )

    return response.choices[0].message.content