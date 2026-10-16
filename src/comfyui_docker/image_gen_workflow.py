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
from typing import Any, Optional

logger = logging.getLogger("image_gen_workflow")
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

from workflows import Workflow, step
from workflows.events import Event, StartEvent, StopEvent
from pydantic import BaseModel, Field


from openai import AsyncOpenAI as OpenAI, BadRequestError

# Define event types for each category
class PromptEvent(StartEvent):
    prompt: str = Field(description="Base Prompt for the workflow")

class PictureStyleEvent(Event):
    """Event containing the picture style prompt component."""
    picture_style: str = Field(description="Visual style, medium, artistic movement, and lighting")


class CharactersEvent(Event):
    """Event containing the characters prompt component."""
    characters: str = Field(description="Character descriptions, appearance, clothing, expressions")


class InteractionsEvent(Event):
    """Event containing the interactions prompt component."""
    characters: list[dict[str, str]] = Field(description="List of characters and descriptions")
    interactions: str = Field(description="Actions, poses, spatial relationships")


class BackgroundEvent(Event):
    """Event containing the background prompt component."""
    background: str = Field(description="Setting, environment, composition, atmospheric details")


class CharacterEvent(Event):
    """Event prompting a character identified in the scene."""
    character: dict[str,str|int] = Field(description="List of character identifiers")


class CharacterDescriptionEvent(Event):
    """Event containing the description for a character."""
    name: str = Field("Character name")
    description: str = Field("Character description")


class CombinedPromptEvent(StopEvent):
    """Event containing the combined prompt from all components."""
    combined_prompt: str = Field(description="Final combined prompt for image generation")


class LLMConfig(BaseModel):
    """Configuration for the LLM client."""
    model: str = Field(default="docker.io/local/gemma-4:latest", description="LLM model identifier")
    base_url: str = Field(default="http://172.17.0.1:12434/engines/llama.cpp/v1/", description="Base URL for the LLM API")
    api_key: str = Field(default="Dummy", description="API key for the LLM service")
    max_tokens: int = Field(default=2048, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.7, description="Temperature for LLM sampling")
    seed: int = Field(default=-1, description="Seed for generating text.")


class ImageConfig(BaseModel):
    """Configuration for image generation."""
    workflow: str = Field(default="workflows/krea_2_turbo.json", description="Path to the ComfyUI workflow JSON")
    prompt_slot: str = Field(default="6.text", description="Slot name for the prompt in the workflow")
    width_slot: str = Field(default="9.width", description="Slot name for width in the workflow")
    height_slot: str = Field(default="9.height", description="Slot name for height in the workflow")
    seed_slot: str = Field(default="8.seed", description="Slot name for seed in the workflow")
    width: int = Field(default=1024, description="Image width in pixels")
    height: int = Field(default=1024, description="Image height in pixels")
    pixels: int = Field(default=2048, description="Total pixel dimension")
    wait: bool = Field(default=False, description="Whether to wait for image generation")
    free_comfyui: bool = Field(default=False, description="Free ComfyUI memory after generation")
    free_model: bool = Field(default=False, description="Free Docker model memory after generation")


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
        timeout: int = 45,
        seed: int = -1
    ):
        super().__init__(timeout=timeout)
        self.llm_config = llm_config
        self.image_config = image_config
        self.user_prompt = user_prompt
        self.system_prompts = system_prompts
        self.seed = seed
        self.image_data = PromptWorkflow._encode_image(image_path) if image_path != None else None
        self.client = OpenAI(
            base_url=self.llm_config.base_url,
            api_key=self.llm_config.api_key,
        )

    @staticmethod
    def _encode_image(image_path: Path) -> str:
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

    async def _call_llm(self, system_prompt: str, user_content: Any, timeout: int = 60) -> str|None:
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
        messages = [{"role": "system", "content": system_prompt},]
        if self.image_data == None :
            messages.append({"role": "user", "content": user_content})
        else :
            messages.append({"role": "user", "content": [
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": {"url": self.image_data}},
            ]})
        logger.debug(f"LLM Call - System: {system_prompt[:100]}...")
        logger.debug(f"LLM Call - User: {str(user_content)[:100]}...")

        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=messages,
                    max_tokens=self.llm_config.max_tokens,
                    temperature=self.llm_config.temperature,
                    seed=self.seed if self.seed != -1 else random.getrandbits(31)
                ),
                timeout=timeout,
            )
            result = response.choices[0].message.content
            logger.debug(f"LLM Response: {result}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"LLM call timed out after {timeout} seconds")
            raise
        except BadRequestError as e:
            logger.error(f"BadRequestError: {e}")
            raise

    @step
    async def generate_picture_style(self, ev: StartEvent) -> PictureStyleEvent:
        """Generate the picture style component of the prompt."""
        system_prompt = self.system_prompts["picture_style"]["content"]
        picture_style = await self._call_llm(system_prompt, self.user_prompt)
        return PictureStyleEvent(picture_style=picture_style)

    @step
    async def generate_interactions(self, evs: list[CharacterDescriptionEvent]) -> InteractionsEvent:
        """Generate the interactions component of the prompt.
        
        This step is conditionally executed - it only generates interactions
        if there are 2 or more characters in the scene. For 0 or 1 characters,
        it returns an empty string, effectively skipping meaningful interaction
        description.
        """
        characters = [{"name": ev.name, "description": ev.description} for ev in evs]
        # Skip interactions if there are 0 or 1 characters
        if len(evs) <= 1:
            logger.info(f"Skipping interactions generation: only {len(evs)} character(s)")
            return InteractionsEvent(characters=characters, interactions="")
        
        system_prompt = self.system_prompts["interactions"]["content"]
        
        interactions = await self._call_llm(system_prompt, f"The image prompt \"{self.user_prompt}\" with the characters{str(characters)}")
        return InteractionsEvent(characters=characters, interactions=interactions)

    @step
    async def generate_background(self, ev: StartEvent) -> BackgroundEvent:
        """Generate the background component of the prompt."""
        system_prompt = self.system_prompts["background"]["content"]
        background = await self._call_llm(system_prompt, self.user_prompt)
        return BackgroundEvent(background=background)

    @step
    async def identify_characters(self, ev: StartEvent) -> list[CharacterEvent]|InteractionsEvent:
        """Identify and list all characters in the scene from the user prompt or image."""
        system_prompt = self.system_prompts["identify_characters"]["content"]

        if self.image_data:
            user_content = [
                {"type": "text", "text": self.user_prompt},
                {"type": "image_url", "image_url": {"url": self.image_data}},
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
        except (json_module.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse character list JSON: {e}")
            logger.error(f"Raw response: {result}")
            characters = []
        if len(characters) > 0:
            return [CharacterEvent(character=character) for character in characters]
        else :
            return InteractionsEvent(characters=[], interactions="")

    @step
    async def generate_character_description(
        self, ev: CharacterEvent
    ) -> CharacterDescriptionEvent:
        """Generate a detailed description for each character individually."""
        system_prompt = self.system_prompts["characters"]["content"]
        character = ev.character
        char_name = character.get('name', f"Character_{hash(str(character)) % 1000}")
                
        description = await self._call_llm(system_prompt, f"Focus on describing this character: {character}. {self.user_prompt}")
        return CharacterDescriptionEvent(name=char_name, description=description)

    @step
    async def combine_prompts(
        self,
        picture_style_ev: PictureStyleEvent,
        interaction_ev: InteractionsEvent,
        background_ev: BackgroundEvent,
    ) -> CombinedPromptEvent:
        """Combine all prompt components into a single cohesive prompt."""
        system_prompt = self.system_prompts["summarizer"]["content"]

        # Create a combined input for the summarizer
        combined_input = f"Picture Style: {picture_style_ev.picture_style}"
        if len(interaction_ev.characters) > 0 :
            combined_input += f"\nCharacter{'s' if len(interaction_ev.characters)>1 else ''}: {str(interaction_ev.characters)}"
        if len(interaction_ev.characters) > 1:
            combined_input += f"\nInteractions: {interaction_ev.interactions}"
        combined_input += f"\nBackground: {background_ev.background}"
        combined_input += "\n\nPlease combine these into a single, cohesive image-generation prompt."

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
    print(combined_prompt)
    return combined_prompt


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
    parser.add_argument("--api-key", "-a", type=str)
    parser.add_argument("--system-prompts", "-s", type=str,
                        help="Path to system_prompts.yaml file")
    parser.add_argument("--max-tokens", "-M", type=int, default=2048)
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--image")
    args = parser.parse_args()

    # Run the async workflow
    asyncio.run(run_workflow(args))


if __name__ == "__main__":
    main()
