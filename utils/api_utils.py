import base64
import os
import requests
import traceback
import time

from openai import OpenAI

from constants import *

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def vlm_generate(
    prompt,
    image,
    key=VLM_API_KEY,
    url=VLM_API_URL, 
    model=VLM_MODEL_TYPE, 
):
    if os.path.exists(image):
        image = f"data:image/jpeg;base64,{encode_image(image)}"

    client = OpenAI(api_key=key, base_url=url)

    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ],
    )
    # print("Chat response:", chat_response.choices[0].message.content)

    return chat_response.choices[0].message.content

def llm_generate(
    prompt, 
    key=LLM_API_KEY,
    url=LLM_API_URL, 
    model=LLM_MODEL_TYPE, 
    max_tokens=8192, 
    temperature=1.0
):
    client = OpenAI(api_key=key, base_url=url)

    res = "None"
    cnt = 0
    while cnt < 20:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant skilled in handling tabular data."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False
            )
            res = response.choices[0].message.content
            break
        except Exception as e:
            print(f"LLM API Request Failed! Retry {cnt}!")
            traceback.print_exc()
            import time; time.sleep(0.1)
            cnt += 1
    
    return res

def main():
    print(llm_generate("Tell something about your!"))
    print(vlm_generate("Tell something about the image!", "/mnt/petrelfs/tangzirui/ST-Raptor/assets/examples.png"))

if __name__ == "__main__":
    main()
