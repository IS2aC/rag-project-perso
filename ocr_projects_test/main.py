import ollama

# image_path = "data_tickets/ticket0.jpg"
image_path = "data_tickets/ticket1.png"


response = ollama.chat(
    model="qwen3-vl:2b",
    messages=[
        {
            "role": "user",
            "content": "Extract all the text and numbers from this image.",
            "images": [image_path]
        }
    ]
)

print("-----------------------------------------------------------------------------------------------")
print("")
print("")
print(response['message']['content'])