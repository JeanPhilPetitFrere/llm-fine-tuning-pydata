import torch
from transformers import BitsAndBytesConfig

# Type to cast the models submodules
compute_dtype = getattr(torch, "float16")

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # loads the model with 4 bits precision Using QLoRA
    bnb_4bit_quant_type="nf4",  # Variant of 4bit configuration other being FP4 -> nf4 is more
    # resilient to temperature variation, better for llama series of model
    bnb_4bit_compute_dtype=compute_dtype,  # Type to cast the models submodules ->     # Submodules allow a module designer to split
    # a complex model into several pieces where all the submodules contribute to a single namespace,
    # which is defined by the module that includes the submodules
    bnb_4bit_use_double_quant=False,  # controls whether we use a second quantization to save an additional 0.4 bits per parameter
)
