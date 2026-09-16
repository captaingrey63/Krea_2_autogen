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
import math

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


def get_image_dimensions(image_path):
    import cv2
    """
    Get the dimensions (width, height) of an image file using OpenCV.

    Args:
        image_path (str): Path to the image file

    Returns:
        tuple: (width, height) of the image
    """
    # Read the image
    image = cv2.imread(image_path)

    # Check if image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    # Get dimensions (height, width, channels)
    height, width = image.shape[:2]

    return width, height

def main():
    parser = ArgumentParser()
    parser.add_argument("PROMPT", nargs="*")
    parser.add_argument("--free-comfyui", "-f", action="store_true", 
                        help="Tell the ComfyUI process to free memory")
    parser.add_argument("--free-model", "-F", action="store_true",
                        help="Tell the Docker Model Runner to free memory")
    parser.add_argument("--model", "-m", default="docker.io/local/gemma-4:latest")
    parser.add_argument("--dmr-base-url", "-e", type=str)
    parser.add_argument("--workflow", "-w", default="workflows/krea_2_turbo.json")
    parser.add_argument("--prompt-slot", "-S", default="6.text")
    parser.add_argument("--width-slot", default="9.width")
    parser.add_argument("--height-slot", default="9.height")
    parser.add_argument("--pixels", type=int, default=2048, help="Width + height for generated image from a supplied image")
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
    width = 1024
    height = 1024
    # Read in the base image if supplied
    if args.image != None :
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"ERROR: can't find {image_path}")
            exit(1)
        mime_type = get_image_mime_type(image_path)
        with open(image_path, "rb") as image_fd:
            b64_image = base64.b64encode(image_fd.read()).decode(encoding="utf-8")
        width, height = get_image_dimensions(image_path)
        scale = float(args.pixels) / float(width + height)
        width = math.floor(float(width) * scale / 16) * 16
        height = math.floor(float(height) * scale / 16) * 16
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
    base_url = os.environ.get("OPENAI_BASE_URL", None) if args.dmr_base_url == None else args.dmr_base_url
    # Default to the Docker Model Runner endpoint
    if base_url == None :
        base_url = "http://172.17.0.1:12434/engines/llama.cpp/v1/"
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
        content = {"role": "user", "content" : []}
        if prompt != None:
            content["content"].append({"type": "text", "text": prompt})
        content["content"].append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}})
        messages.append(content)

    logger.debug(json.dumps(messages, indent=2))

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
    cmd = f"uv run comfy workflow set-slot {args.workflow}".split() + [f"{args.prompt_slot}={response.choices[0].message.content}"]
    if b64_image != None :
        cmd = cmd + [f"{args.width_slot}={width}", f"{args.height_slot}={height}"]
    logger.debug(f"Prompt edit {cmd}")
    proc = subprocess.run(cmd )
    if proc.returncode != 0:
        logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
        exit(proc.returncode)
    logger.info("Queuing ComfyUI workflow")
    cmd = f"uv run comfy run --workflow {args.workflow}".split()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
        exit(proc.returncode)
    
if __name__ == "__main__":
    main()