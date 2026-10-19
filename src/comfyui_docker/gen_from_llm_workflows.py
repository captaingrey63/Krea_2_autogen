#!/usr/bin/env python3
"""
Generate ComfyUI prompts using LlamaIndex Workflows.

This script makes separate LLM calls for each prompt category (picture style, characters,
interactions, background) and combines them using a summarization step.
"""

import asyncio
import json
import yaml
import logging
import os
import subprocess
import sys
import random
from argparse import ArgumentParser
from importlib.resources import files as package_files
from pathlib import Path
from typing import Any

from .image_gen_workflow import LLMConfig, ImageConfig, PromptWorkflow, CombinedPromptEvent
logger = logging.getLogger("gen_from_llm_workflows")
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


async def run_workflow(args: Any) -> str:
    """Run the prompt generation workflow and return the combined prompt."""
    # Load system prompts from YAML file
    if args.system_prompts:
        prompts_file = Path(args.system_prompts)
    else:
        prompts_file = package_files("comfyui_docker.data").joinpath("system_prompts.yaml")

    if not prompts_file.is_file():
        logger.error(f"ERROR: {prompts_file} does not exist")
        sys.exit(1)

    system_prompts = yaml.safe_load(prompts_file.read_text(encoding="utf-8"))

    
    # Load character prompts from YAML file
    character_prompts_file = package_files("comfyui_docker.data").joinpath("character_prompts.yaml")
    if character_prompts_file.is_file():
        character_prompts = yaml.safe_load(character_prompts_file.read_text(encoding="utf-8"))
    else:
        character_prompts = None
    
    # Set up LLM config
    api_key = os.environ.get("OPENAI_API_KEY", None) if args.api_key is None else args.api_key
    if api_key is None:
        api_key = "Dummy"

    base_url = os.environ.get("OPENAI_BASE_URL", None) if args.dmr_base_url is None else args.dmr_base_url
    if base_url is None:
        base_url = "http://172.17.0.1:12434/engines/llama.cpp/v1/"

    llm_config = LLMConfig(
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Set up image config
    image_config = ImageConfig(
        workflow=args.workflow,
        prompt_slot=args.prompt_slot,
        width_slot=args.width_slot,
        height_slot=args.height_slot,
        seed_slot=args.seed_slot,
        width=args.width,
        height=args.height,
        pixels=args.pixels,
        wait=args.wait,
        free_comfyui=args.free_comfyui,
        free_model=args.free_model,
    )

    # Determine user prompt and image
    user_prompt = " ".join(args.PROMPT) if args.PROMPT else None
    image_path = Path(args.image) if args.image else None

    if user_prompt is None and image_path is None:
        logger.error("ERROR: Must supply at least one of a prompt or image")
        sys.exit(1)

    if user_prompt is None:
        user_prompt = ""

    # Free ComfyUI memory if requested
    if args.free_comfyui:
        logger.info("Freeing ComfyUI Memory Use")
        cmd = "uv run comfy free --free-memory".split()
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error(f"ERROR: command {' '.join(cmd)} returned {proc.returncode}")
            sys.exit(proc.returncode)

    # Create and run the workflow
    workflow = PromptWorkflow(
        llm_config=llm_config,
        image_config=image_config,
        user_prompt=user_prompt,
        system_prompts=system_prompts,
        character_prompts=character_prompts,
        image_path=image_path,
        timeout=120
    )

    # Run the workflow
    handler = workflow.run(timeout=600)
    combined_event = None

    async for event in handler.stream_events():
        logger.debug(f"Workflow event: {event}")

    result = await handler
    logger.info(f"Workflow result: {result}")

    # Extract the combined prompt from the result
    if isinstance(result, CombinedPromptEvent):
        combined_prompt = result.combined_prompt
    else:
        logger.error(f"Unexpected result type: {type(result)}")
        sys.exit(1)

    # Free Docker Model memory if requested
    if args.free_model:
        logger.info("Freeing Docker Model memory")
        cmd = "docker model unload --all".split()
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            logger.error(f"ERROR: command {' '.join(cmd)} returned {proc.returncode}")
            sys.exit(proc.returncode)

    # Queue ComfyUI workflow
    if image_path:
        img_width, img_height = get_image_dimensions(image_path)
        new_width, new_height = scale_dimensions(img_width, img_height, args.pixels)
    else:
        new_width, new_height = scale_dimensions(
            args.width if args.width is not None else image_config.width,
            args.height if args.height is not None else image_config.height,
            args.pixels,
        )
    logger.info(f"Using Image Prompt:\n{combined_prompt}")
    cmd = f"uv run comfy workflow set-slot {args.workflow}".split() + [
        f"{args.prompt_slot}={combined_prompt}",
        f"{args.width_slot}={new_width}",
        f"{args.height_slot}={new_height}",
        f"{args.seed_slot}={args.seed if args.seed is not None else random.getrandbits(31)}",
    ]
    logger.debug(f"Prompt edit {cmd}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        logger.error(f"ERROR: command {' '.join(cmd)} returned {proc.returncode}")
        sys.exit(proc.returncode)

    logger.info(f"Queuing ComfyUI workflow")
    cmd = f"uv run comfy run --workflow {args.workflow}".split()
    if args.wait:
        cmd.append("--wait")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        logger.error(f"ERROR: command {' '.join(cmd)} returned {proc.returncode}")
        sys.exit(proc.returncode)

    return combined_prompt


def get_image_dimensions(image_path: str) -> tuple[int, int]:
    """Get the dimensions (width, height) of an image file using OpenCV."""
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")

    height, width = image.shape[:2]
    return width, height


def scale_dimensions(width: int, height: int, pixels: int) -> tuple[int, int]:
    """Scale dimensions to maintain aspect ratio while targeting a pixel count."""
    import math

    scale = float(pixels) / float(width + height)
    new_width = math.floor(float(width) * scale / 16) * 16
    new_height = math.floor(float(height) * scale / 16) * 16
    return new_width, new_height


def main():
    """Main entry point for the script."""

    parser = ArgumentParser()
    parser.add_argument("PROMPT", nargs="*")
    parser.add_argument("--free-comfyui", "-f", action="store_false",
                        help="Tell the ComfyUI process to free memory")
    parser.add_argument("--free-model", "-F", action="store_false",
                        help="Tell the Docker Model Runner to free memory")
    parser.add_argument("--model", "-m", default="docker.io/local/gemma:E2B-Q4_k_m")
    parser.add_argument("--dmr-base-url", "-e", type=str)
    parser.add_argument("--workflow", "-w", default="workflows/krea_2_turbo.json")
    parser.add_argument("--prompt-slot", "-S", default="6.text")
    parser.add_argument("--width-slot", default="9.width")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height-slot", default="9.height")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--pixels", type=int, default=2048,
                        help="Width + height for generated image from a supplied image")
    parser.add_argument("--api-key", "-a", type=str)
    parser.add_argument("--system-prompts", "-s", type=str,
                        help="Path to system_prompts.yaml file")
    parser.add_argument("--max-tokens", "-M", type=int, default=2048)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--image")
    parser.add_argument("--seed-slot", default="8.seed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--wait", action="store_false")
    parser.add_argument("--list-characters", "-l", action="store_true",
                        help="List character names from character_prompts.yaml and exit")
    args = parser.parse_args()

    # Handle --list-characters flag
    if args.list_characters:
        character_prompts_file = package_files("comfyui_docker.data").joinpath("character_prompts.yaml")
        if character_prompts_file.is_file():
            character_prompts = yaml.safe_load(character_prompts_file.read_text(encoding="utf-8"))
            if character_prompts:
                print("Available characters:")
                for name in character_prompts.keys():
                    print(f"  - {name}")
            else:
                print("No characters found in character_prompts.yaml")
        else:
            print("ERROR: character_prompts.yaml not found")
            sys.exit(1)
        sys.exit(0)

    # Run the async workflow
    asyncio.run(run_workflow(args))


if __name__ == "__main__":
    main()
