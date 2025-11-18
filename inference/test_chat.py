from .generate import generate

while True:
    prompt = input("User: ")
    if prompt.lower() in ['exit', 'quit']:
        break
    response = generate(prompt)
    print("Bot:", response)
