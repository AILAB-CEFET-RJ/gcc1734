import ollama

response = ollama.chat(
    model='gemma3',
    messages=[
        {
        'role': 'user',
        'content': 'Translate this sentence to Portuguese: The book is on the table.',
        },
    ],
)

print(response['message']['content'])