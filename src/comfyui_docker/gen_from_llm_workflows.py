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
from typing import Any, Optional, Dict

logger = logging.getLogger("gen_from_llm_workflows")
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# Add the .venv site-packages to the path
venv_path = Path("/home/sdavis/comfyui-docker/.venv/lib/python3.14/site-packages")
if venv_path.exists():
    sys.path.insert(0, str(venv_path))

from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from pydantic import Field

# Import OpenAI client
try:
    from openai import AsyncOpenAI as OpenAI, BadRequestError, InternalServerError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Define event types for each category
class PictureStyleEvent(Event):
    """Event containing the picture style prompt component."""
    picture_style: str = Field(description="Visual style, medium, artistic movement, and lighting")


class CharactersEvent(Event):
    """Event containing the characters prompt component."""
    characters: str = Field(description="Character descriptions, appearance, clothing, expressions")


class InteractionsEvent(Event):
    """Event containing the interactions prompt component."""
    interactions: str = Field(description="Actions, poses, spatial relationships")


class BackgroundEvent(Event):
    """Event containing the background prompt component."""
    background: str = Field(description="Setting, environment, composition, atmospheric details")


class CharacterListEvent(Event):
    """Event containing the list of characters identified in the scene."""
    characters: list[dict[str,str|int]] = Field(description="List of character identifiers")


class CharacterDescriptionsEvent(Event):
    """Event containing descriptions for each character."""
    character_descriptions: Dict[str, str] = Field(
        description="Dictionary mapping character names to their individual descriptions"
    )


class CombinedPromptEvent(StopEvent):
    """Event containing the combined prompt from all components."""
    combined_prompt: str = Field(description="Final combined prompt for image generation")


class LLMConfig:
    """Configuration for the LLM client."""
    def __init__(
        self,
        model: str = "docker.io/local/gemma-4:latest",
        base_url: str = "http://172.17.0.1:12434/engines/llama.cpp/v1/",
        api_key: str = "Dummy",
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature


class ImageConfig:
    """Configuration for image generation."""
    def __init__(
        self,
        workflow: str = "workflows/krea_2_turbo.json",
        prompt_slot: str = "6.text",
        width_slot: str = "9.width",
        height_slot: str = "9.height",
        seed_slot: str = "8.seed",
        width: int = 1024,
        height: int = 1024,
        pixels: int = 2048,
        wait: bool = False,
        free_comfyui: bool = False,
        free_model: bool = False,
    ):
        self.workflow = workflow
        self.prompt_slot = prompt_slot
        self.width_slot = width_slot
        self.height_slot = height_slot
        self.seed_slot = seed_slot
        self.width = width
        self.height = height
        self.pixels = pixels
        self.wait = wait
        self.free_comfyui = free_comfyui
        self.free_model = free_model


class PromptWorkflow(Workflow):
    """
    Workflow that generates image prompts by querying the LLM separately
    for each category (picture style, characters, interactions, background)
    and then combining them.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        image_config: ImageConfig,
        user_prompt: str,
        system_prompts: dict[str, dict[str, str]],
        image_path: Optional[Path] = None,
    ):
        super().__init__()
        self.llm_config = llm_config
        self.image_config = image_config
        self.user_prompt = user_prompt
        self.system_prompts = system_prompts
        self.image_path = image_path
        self.client = self._create_llm_client()

    def _create_llm_client(self) -> OpenAI:
        """Create an OpenAI client with the configured settings."""
        return OpenAI(
            base_url=self.llm_config.base_url,
            api_key=self.llm_config.api_key,
        )

    def _encode_image(self, image_path: Path) -> str:
        """Encode an image to base64 data URL."""
        import base64

        # Get MIME type from file extension (replaces deprecated imghdr)
        ext = image_path.suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp',
        }
        mime_type = mime_types.get(ext, 'image/png')

        with open(image_path, "rb") as image_fd:
            b64_image = base64.b64encode(image_fd.read()).decode(encoding="utf-8")

        return f"data:{mime_type};base64,{b64_image}"

    async def _call_llm(self, system_prompt: str, user_content: Any, timeout: int = 60) -> str:
        """Make a call to the LLM with the given system prompt and user content.
        
        Args:
            system_prompt: The system prompt for the LLM
            user_content: The user content for the LLM
            timeout: Timeout in seconds for the LLM call (default: 60)
            
        Returns:
            The LLM response text
            
        Raises:
            asyncio.TimeoutError: If the LLM call takes longer than timeout seconds
            BadRequestError: If there's a bad request error
            InternalServerError: If there's an internal server error
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.debug(f"LLM Call - System: {system_prompt[:100]}...")
        logger.debug(f"LLM Call - User: {str(user_content)[:100]}...")

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=messages,
                    max_tokens=self.llm_config.max_tokens,
                    temperature=self.llm_config.temperature,
                ),
                timeout=timeout,
            )
            result = response.choices[0].message.content
            logger.info(f"LLM Response: {result}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {timeout} seconds")
            raise
        except BadRequestError as e:
            logger.error(f"BadRequestError: {e}")
            raise
        except InternalServerError as e:
            logger.error(f"InternalServerError: {e}")
            raise

    @step
    async def generate_picture_style(self, ev: StartEvent) -> PictureStyleEvent:
        """Generate the picture style component of the prompt."""
        system_prompt = self.system_prompts["picture_style"]["content"]
        
        if self.image_path:
            image_data_url = self._encode_image(self.image_path)
            user_content = [
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            user_content = self.user_prompt

        picture_style = await self._call_llm(system_prompt, user_content)
        return PictureStyleEvent(picture_style=picture_style)

    @step
    async def generate_interactions(self, ev: CharacterListEvent) -> InteractionsEvent:
        """Generate the interactions component of the prompt.
        
        This step is conditionally executed - it only generates interactions
        if there are 2 or more characters in the scene. For 0 or 1 characters,
        it returns an empty string, effectively skipping meaningful interaction
        description.
        """
        # Skip interactions if there are 0 or 1 characters
        if len(ev.characters) <= 1:
            logger.info(f"Skipping interactions generation: only {len(ev.characters)} character(s)")
            return InteractionsEvent(interactions="")
        
        system_prompt = self.system_prompts["interactions"]["content"]

        if self.image_path:
            image_data_url = self._encode_image(self.image_path)
            user_content = [
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            user_content = self.user_prompt

        interactions = await self._call_llm(system_prompt, user_content)
        return InteractionsEvent(interactions=interactions)

    @step
    async def generate_background(self, ev: StartEvent) -> BackgroundEvent:
        """Generate the background component of the prompt."""
        system_prompt = self.system_prompts["background"]["content"]

        if self.image_path:
            image_data_url = self._encode_image(self.image_path)
            user_content = [
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            user_content = self.user_prompt

        background = await self._call_llm(system_prompt, user_content)
        return BackgroundEvent(background=background)

    @step
    async def identify_characters(self, ev: StartEvent) -> CharacterListEvent:
        """Identify and list all characters in the scene from the user prompt or image."""
        system_prompt = self.system_prompts["identify_characters"]["content"]

        if self.image_path:
            image_data_url = self._encode_image(self.image_path)
            user_content = [
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        else:
            user_content = self.user_prompt

        result = await self._call_llm(system_prompt, user_content)
        
        # Parse the JSON response
        import json as json_module
        try:
            # Try to extract JSON array from the result
            # Handle potential markdown code blocks
            result_clean = result.strip()
            if result_clean.startswith('```') and result_clean.endswith('```'):
                result_clean = result_clean[3:-3].strip()
            if result_clean.startswith('json'):
                result_clean = result_clean[4:].strip()
            
            characters = json_module.loads(result_clean)
            if not isinstance(characters, list):
                logger.warning(f"Expected JSON array, got: {result_clean}")
                characters = []
            return CharacterListEvent(characters=characters)
        except (json_module.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse character list JSON: {e}")
            logger.error(f"Raw response: {result}")
            # Fallback: try to extract names manually
            return CharacterListEvent(characters=[])

    @step
    async def generate_individual_character_descriptions(
        self, ev: CharacterListEvent
    ) -> CharacterDescriptionsEvent:
        """Generate a detailed description for each character individually.
        
        This method uses parallel execution to generate descriptions for all
        characters concurrently, which significantly reduces the total time
        compared to sequential execution. It also pre-encodes the image once
        to avoid redundant encoding for each character.
        
        Args:
            ev: CharacterListEvent containing the list of characters to describe
            
        Returns:
            CharacterDescriptionsEvent with a dictionary mapping character names
            to their individual descriptions
        """
        if not ev.characters:
            logger.info("No characters to describe, returning empty descriptions")
            return CharacterDescriptionsEvent(character_descriptions={})
        
        system_prompt = self.system_prompts["characters"]["content"]
        
        # Pre-encode image once if available
        image_data_url = None
        if self.image_path:
            image_data_url = self._encode_image(self.image_path)
        
        # Calculate per-character timeout based on total timeout and number of characters
        # Leave room for overhead (20% of total timeout)
        per_character_timeout = max(10, 20 * len(ev.characters))  # At least 10s, scale with count
        
        logger.info(f"Generating descriptions for {len(ev.characters)} characters with timeout {per_character_timeout}s each")
        
        async def describe_character(character: dict[str, Any]) -> tuple[str, str]:
            """Generate a description for a single character."""
            try:
                char_name = character.get('name', f"Character_{hash(str(character)) % 1000}")
                
                # Create character-specific user content
                if image_data_url:
                    user_content = [
                        {"type": "text", "text": f"Focus on describing this character: {character}. {self.user_prompt}"},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ]
                else:
                    user_content = f"Focus on describing this character: {character}. {self.user_prompt}"
                
                description = await self._call_llm(system_prompt, user_content, timeout=per_character_timeout)
                return char_name, description
            except asyncio.TimeoutError:
                logger.warning(f"Timeout generating description for character: {character.get('name', 'unknown')}")
                return character.get('name', f"Character_{hash(str(character)) % 1000}"), f"Description timed out for {character.get('name', 'this character')}"
            except Exception as e:
                logger.error(f"Error generating description for character {character.get('name', 'unknown')}: {e}")
                return character.get('name', f"Character_{hash(str(character)) % 1000}"), f"Error generating description for {character.get('name', 'this character')}"
        
        # Run all character descriptions in parallel
        description_tasks = [describe_character(char) for char in ev.characters]
        results = await asyncio.gather(*description_tasks, return_exceptions=False)
        
        # Build the character descriptions dictionary
        character_descriptions = {name: desc for name, desc in results}
        
        return CharacterDescriptionsEvent(character_descriptions=character_descriptions)

    @step
    async def combine_prompts(
        self,
        picture_style_ev: PictureStyleEvent,
        character_descriptions_ev: CharacterDescriptionsEvent,
        background_ev: BackgroundEvent,
        interactions_ev: InteractionsEvent,
    ) -> CombinedPromptEvent:
        """Combine all prompt components into a single cohesive prompt."""
        system_prompt = self.system_prompts["summarizer"]["content"]

        # Format character descriptions
        characters_text = "\n".join(
            f"{name}: {desc}" for name, desc in character_descriptions_ev.character_descriptions.items()
        )
        
        # Format interactions (may be empty if 0 or 1 characters)
        interactions_text = interactions_ev.interactions if interactions_ev.interactions else "No specific interactions"

        # Create a combined input for the summarizer
        combined_input = f"""Picture Style: {picture_style_ev.picture_style}
Characters: {characters_text}
Interactions: {interactions_text}
Background: {background_ev.background}

Please combine these into a single, cohesive image-generation prompt."""

        combined_prompt = await self._call_llm(system_prompt, combined_input)
        return CombinedPromptEvent(combined_prompt=combined_prompt)


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
        image_path=image_path,
    )

    # Run the workflow
    handler = workflow.run(timeout=120)
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
    args = parser.parse_args()

    # Run the async workflow
    asyncio.run(run_workflow(args))


if __name__ == "__main__":
    main()
