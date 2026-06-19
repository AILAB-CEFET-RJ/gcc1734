import ollama
import chainlit as cl

@cl.on_message
async def main(message: cl.Message):
    response = ollama.chat(
        model='gemma3',
        messages=[
            {
            'role': 'user',
            'content': message.content,
            },
        ],
    )
    
    await cl.Message(content=response['message']['content']).send()

@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! I'm Gemma3. How can I help you today?").send()