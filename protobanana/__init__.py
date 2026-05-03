"""protoBanana — OSS chat-native image generation + editing on the protoLabs gateway.

Wraps a ComfyUI backend (Qwen-Image-2512 + Qwen-Image-Edit-2511 + Florence-2 +
SAM 2.1 + BiRefNet/RMBG) as an OpenAI-compatible LiteLLM custom provider. One
gateway alias drives text-to-image, instruction-edit, multi-reference compose,
background removal, region-aware edit, inpaint, and outpaint — picked per turn
from a chat-completions message stream.

The published artifact is intended to be `pip install`-able into a LiteLLM
container. See README.md for quickstart and PHASES.md for the roadmap.
"""

from protobanana.client import ComfyUIClient
from protobanana.intents.keywords import (
    Operation,
    classify_operation,
    infer_size_from_prompt,
)
from protobanana.provider import ProtoBananaProvider, handler
from protobanana.workflows.loader import WorkflowLoader

__version__ = "0.1.0a0"

__all__ = [
    "ProtoBananaProvider",
    "ComfyUIClient",
    "Operation",
    "WorkflowLoader",
    "classify_operation",
    "infer_size_from_prompt",
    "handler",
    "__version__",
]
