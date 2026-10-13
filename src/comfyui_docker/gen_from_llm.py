import os
import subprocess
from pathlib import Path
from importlib.resources import files as package_files
from argparse import ArgumentParser
from openai import OpenAI, BadRequestError, InternalServerError
import base64
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
    """Detect image MIME type from file extension.
    
    Replaces the deprecated imghdr module.
    """
    ext = file_path.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.webp': 'image/webp',
    }
    return mime_types.get(ext, 'image/png')


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
def scale_dimensions(width:int, height:int, pixels:int) -> tuple[int,int] :
    scale = float(pixels) / float(width + height)
    new_width = math.floor(float(width) * scale / 16) * 16
    new_height = math.floor(float(height) * scale / 16) * 16
    return new_width, new_height

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
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height-slot", default="9.height")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--pixels", type=int, default=2048, help="Width + height for generated image from a supplied image")
    parser.add_argument("--api-key", "-a", type=str)
    parser.add_argument("--system-prompt", "-s", type=str)
    parser.add_argument("--max-tokens", "-M", type=int, default=2048)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--image")
    parser.add_argument("--seed-slot", default="8.seed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--wait", action="store_true")
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
    images = []
    # Read in the base image(s) if supplied
    if args.image != None :
        image_path = Path(args.image)
        if image_path.is_file() :
            images.append({"image_path":image_path})
        elif image_path.is_dir() :
            image_files = list(image_path.glob("*.png", case_sensitive=False)) + list(image_path.glob("*.jpg", case_sensitive=False)) + list(image_path.glob("*.jpeg", case_sensitive=False))
            for image in image_files :
                images.append({"image_path":image})
        else:
            print(f"ERROR: can't find an image at {image_path}")
            exit(1)
        for i, _ in enumerate(images) :
            width, height = get_image_dimensions(images[i]['image_path'])
            images[i]['width'], images[i]['height'] = \
                scale_dimensions(width, height, args.pixels)
    else :
        new_width, new_height = \
            scale_dimensions(args.width if args.width != None else width, args.height if args.height != None else height, args.pixels)
        images.append({"image_path":None, "width": new_width, "height": new_height})
    logger.info(images)
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
    for i, _ in enumerate(images) :
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
            images[i]['base_prompt'] = response.choices[0].message.content
        elif args.PROMPT != None and len(args.PROMPT) > 0 :
            images[i]['base_prompt'] = ' '.join(args.PROMPT)
        else : 
            images[i]['base_prompt'] = None
        # Construct the LLM Prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": None}
        ]
        if images[i]['base_prompt'] == None and images[i]['image_path'] == None :
            print("ERROR: Must supply at least one of a prompt or image")
            exit(1)
        elif images[i]['image_path'] == None :
            messages[1]["content"] = images[i]['base_prompt']
        else :
            messages[1]["content"] = []
            if images[i]['base_prompt'] != None :
                messages[1]["content"].append({"type": "text", "text": images[i]['base_prompt']})
            mime_type = get_image_mime_type(images[i]['image_path'])
            with open(images[i]['image_path'], "rb") as image_fd :
                b64_image = base64.b64encode(image_fd.read()).decode(encoding="utf-8")
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}})

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
        logger.info(f"Response for {images[i]['image_path']}\n{response.choices[0].message.content}\n")
        if len(response.choices[0].message.content) < 10 :
            logger.error("LLM response too short")
            exit(1)
        images[i]["positive_prompt"] = response.choices[0].message.content

    # Done with the LLM, free from memory if needed
    if args.free_model :
        logger.info("Freeing Docker Model memory")
        cmd = "docker model unload --all".split()
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
            exit(proc.returncode)
    for image in images :
        cmd = f"uv run comfy workflow set-slot {args.workflow}".split() + [f"{args.prompt_slot}={image['positive_prompt']}", f"{args.width_slot}={image['width']}", f"{args.height_slot}={image['height']}", f"{args.seed_slot}={args.seed if args.seed != None else random.getrandbits(31)}"]
        logger.debug(f"Prompt edit {cmd}")
        proc = subprocess.run(cmd )
        if proc.returncode != 0:
            logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
            exit(proc.returncode)
        logger.info(f"Queuing ComfyUI workflow for {images[i]['image_path']}")
        cmd = f"uv run comfy run --workflow {args.workflow}".split()
        if args.wait == True :
            cmd.append("--wait")
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
           logger.error(f"ERROR: command \"{' '.join(cmd)}\" returned {proc.returncode}")
           exit(proc.returncode)
    
if __name__ == "__main__":
    main()