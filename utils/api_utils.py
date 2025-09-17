import base64
import os
import requests

from openai import OpenAI

from utils.constants import *


def generate_text_third_party(
    prompt, key, url, model="gpt-4o-2024-11-20", max_tokens=512, temperature=1.0
):

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant skilled in handling tabular data.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "n": 1,  # 生成一条回复
        "stop": None,
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": key,
        "content-type": "application/json",
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    print("Chat response:", response.json()["choices"][0]["message"]["content"].strip())

    return response.json()["choices"][0]["message"]["content"].strip()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def llm_generate(prompt, model=LLM_MODEL_TYPE, port=LLM_PORT):
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = f"http://localhost:{port}/v1"

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    # print("Chat response:", chat_response.choices[0].message.content)

    return chat_response.choices[0].message.content


def vlm_generate(
    prompt="Describe this image in one sentence.",
    image="https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
    model=VLM_MODEL_TYPE,
    port=VLM_PORT,
):
    """_summary_

    Args:
        prompt (str, optional): _description_. Defaults to 'Describe this image in one sentence.'.
        image (str, optional): url or image_path. Defaults to 'https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg'.
    """
    if os.path.exists(image):
        image = f"data:image/jpeg;base64,{encode_image(image)}"
    # Set OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = #在此处填写你的key   
    openai_api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"#这里默认使用qwen-vl-max，也可改成你喜欢的模型url

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="qwen-vl-max",
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


def generate_deepseek(
    prompt, key=API_KEY, url=API_URL, model=LLM_MODEL_TYPE, max_tokens=8192, temperature=1.0
):
    client = OpenAI(api_key=key, base_url=url)

    res = ""
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

        except Exception as e:
            print(e)
            print(f"Deepseek API 请求失败！尝试第{cnt}次！")
            import traceback; traceback.print_exc()
            import time; time.sleep(0.1)
            cnt += 1

        break
    
    return res

def generate_deepseek_old(
    prompt, key=API_KEY, url=API_URL, model=LLM_MODEL_TYPE, max_tokens=20480, temperature=1.0
):
    # DeepSeek-R1:671B
    # deepseek-v3:671b

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant skilled in handling tabular data.",
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "n": 1,
        "stop": None,
        "temperature": temperature,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": key,
    }

    res = ""
    cnt = 0
    while cnt < 100:
        try:
            response = requests.request("POST", url, json=payload, headers=headers)
        
            # print("Chat response:", response.json()['choices'][0]['message']['content'].strip())

            rj = response.json()
            if 'message' in rj and "context length" in rj['message']:
                print(rj)
                print(prompt)
                return "0"

            res = response.json()["choices"][0]["message"]["content"].strip()

            return res
            
        except Exception as e:
            print(e)
            print(f"Deepseek API 请求失败！尝试第{cnt}次！")
            import traceback; traceback.print_exc()
            import time; time.sleep(0.1)

        cnt += 1

    return res

def main():
    print(generate_deepseek("hello!",key=API_KEY, url=API_URL, model = 'qwen'))


if __name__ == "__main__":
    main()
