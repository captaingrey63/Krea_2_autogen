import os
import subprocess
from pathlib import Path
from importlib.resources import files as package_files
from argparse import ArgumentParser
from openai import OpenAI, BadRequestError, InternalServerError
import base64
import imghdr
import json
from typing import Dict, List, Any
import random
import logging
logger = logging.getLogger("gen_from_llm")
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
def get_image_mime_type(file_path:Path):
    """Detect image MIME type from file path using imghdr."""
    mime_type = imghdr.what(file_path)
    if mime_type:
        return f"image/{mime_type}"
    return None

def print_truncated_messages(messages:List[Dict[str,str|Dict[str,str]]]) :
    truncated_messages = []
    for message in messages:
        truncated_message = dict()
        for (key, item) in message.items():
            if type(item) == str :
                truncated_message[key] = (item[:20] + ' ...') if len(item) > 20 else item
            elif type(item) == dict :
                trunc_item = dict()
                for (sub_key, sub_item) in item.items():
                    trunc_item[sub_key] = (sub_item[0:20] + ' ...') if len(sub_item) > 20 else sub_item
                truncated_message[key] = trunc_item
        truncated_messages.append(truncated_message)
    logger.debug(json.dumps(truncated_messages, indent=2))
def main():
    parser = ArgumentParser()
    parser.add_argument("PROMPT", nargs="*")
    parser.add_argument("--free-comfyui", "-f", action="store_true", 
                        help="Tell the ComfyUI process to free memory")
    parser.add_argument("--free-model", "-F", action="store_true",
                        help="Tell the Docker Model Runner to free memory")
    parser.add_argument("--model", "-m", default="docker.io/local/gemma-4:latest")
    parser.add_argument("--openai-base-url", "-e", type=str)
    parser.add_argument("--api-key", "-a", type=str)
    parser.add_argument("--system-prompt", "-s", type=str)
    parser.add_argument("--max-tokens", "-M", type=int, default=2048)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--image")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    # Read in the system prompt
    if args.system_prompt != None :
        prompt_file = Path(args.system_prompt)
    else :
        prompt_file = package_files("comfyui_docker.data").joinpath("system_prompt.txt")
    if not prompt_file.is_file():
        print(f"ERROR, f{prompt_file} does not exist")
        exit(1)
    system_prompt = prompt_file.read_text(encoding="utf-8")

    # Read in the base image if supplied
    if args.image != None :
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"ERROR: can't find {image_path}")
            exit(1)
        mime_type = get_image_mime_type(image_path)
        with open(image_path, "rb") as image_fd:
            b64_image = base64.b64encode(image_fd.read()).decode(encoding="utf-8")
    else :
        b64_image = None
    if args.free_comfyui :
        logger.info("Freeing ComfyUI Memory Use")
        cmd = "uv run comfy free --free-memory".split()
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
            exit(proc.returncode)
    api_key = os.environ.get("OPENAI_API_KEY", None) if args.api_key == None else args.api_key
    if api_key == None:
        api_key = "Dummy"
    base_url = os.environ.get("OPENAI_BASE_URL", None) if args.openai_base_url == None else args.openai_base_url
    # Default to the Docker Model Runner endpoint
    if base_url == None :
        base_url = "http://172.17.0.1:12434/v1"
    client = OpenAI(base_url = base_url,
                    api_key = api_key)
    # Gather the user prompt if supplied
    if args.random == True:
        gender = random.choice(package_files("comfyui_docker.data").joinpath("gender.txt").read_text().splitlines())
        character_type = random.choice(package_files("comfyui_docker.data").joinpath("character_type.txt").read_text().splitlines())
        style = random.choice(package_files("comfyui_docker.data").joinpath("style.txt").read_text().splitlines())
        messages = [
                {"role": "system", "content": "Make a one sentence description with the age, gender, general description of significant features, clothing and background. Include a style of image to generate. Only output the sentence, no formatting."},
                {"role": "user", "content": f"An image in the style of {style} of a {gender} {character_type} of a random age over 18 in a random setting and random pose"}
            ]
        try :
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except BadRequestError as e :
            print (f"{e}")
            exit(1)
        except InternalServerError as e :
            print (f"{e}")
            exit(1)
        logger.debug(response)
        logger.info(f"Response\n{response.choices[0].message.content}")
        if len(response.choices[0].message.content) < 10 :
            logger.error("LLM response too short")
            exit(1)
        prompt = response.choices[0].message.content
    elif args.PROMPT != None and len(args.PROMPT) > 0 :
        prompt = ' '.join(args.PROMPT)
    else : 
        prompt = None
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    if b64_image == None :
        if prompt == None:
            print("ERROR: Must supply at least one of a prompt or image")
            exit(1)
        messages.append({"role": "user", "content": prompt})
    if b64_image != None :
        messages.append({"role": "user", "content" : []})
        if prompt != None:
            messages[-1]["content"].append({"type": "text", "text": prompt})
        messages[-1]["content"].append({"type": "image_url", "image_url": f"data:{mime_type};base64,{b64_image}"})

    print_truncated_messages(messages)
    try :
        response = client.chat.completions.create(
            model=args.model,
            messages=messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    except BadRequestError as e :
        print (f"{e}")
        exit(1)
    except InternalServerError as e :
        print (f"{e}")
        exit(1)
    logger.debug(response)
    logger.info(f"Response\n{response.choices[0].message.content}\n")
    if len(response.choices[0].message.content) < 10 :
        logger.error("LLM response too short")
        exit(1)
    if args.free_model :
        logger.info("Freeing Docker Model memory")
        cmd = "docker model unload --all".split()
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
            exit(proc.returncode)
    cmd = "uv run comfy workflow set-slot workflows/krea_2_turbo.json".split()
    proc = subprocess.run(cmd + [f"6.text={response.choices[0].message.content}"])
    if proc.returncode != 0:
        logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
        exit(proc.returncode)
    logger.info("Queuing ComfyUI workflow")
    cmd = "uv run comfy run --workflow workflows/krea_3_lora.json".split()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
        exit(proc.returncode)
    
if __name__ == "__main__":
    main()