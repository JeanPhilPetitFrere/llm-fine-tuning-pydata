import re

from peft import LoraConfig
from transformers import AutoModelForCausalLM

lora_alpha = (
    16,
)  # when optimizing with adam: tuning this is roughly the same as tuning the learning rate
# if the initialization was scaled properly

lora_dropout = (
    0.1,
)  # standard dropout: reduce overfitting by randomly selecting neurons to ignore with a dropout probability during training

r = (
    64,
)  # represents the rank of the low rank matrices learned during the finetuning process.
# As this value is increased, the number of parameters needed to be updated during the low-rank adaptation increases.
# a lower r may lead to a quicker, less computationally intensive training process, but may affect the quality of the model thus produced.
# However, increasing r beyond a certain value may not yield any discernible increase in quality of model output

bias = (
    "none",
)  # can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training.
# Even when disabling the adapters, the model will not produce the same output as the base model would have without adaptation. The default is None

task_type = "CAUSAL_LM"


def get_target_modules(model: AutoModelForCausalLM) -> list:
    """
    Search the llm architecture for all of the linear layers, so that we can update them during fine-tuning

    Parameters
    ----------
    model : AutoModelForCausalLM
        The LLM

    Returns
    -------
    target_modules : list
        list of all of the linear layers in the LLM architecture
    """
    model_modules = str(model.modules)
    pattern = r"\((\w+)\): Linear"
    linear_layer_names = re.findall(pattern, model_modules)

    names = []
    # Print the names of the Linear layers
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))
    return target_modules


def get_qlora_config(
    model: AutoModelForCausalLM,
    lora_alpha: int = lora_alpha,
    lora_dropout: float = lora_dropout,
    r: int = r,
    bias: str = bias,
    task_type: str = task_type,
) -> LoraConfig:
    """Config for the QLoRA fine-tuning

    Parameters
    ----------
    model : AutoModelForCausalLM
        LLM
    target_modules : list
        All of the linear layers prensent in the LLM
    lora_alpha : int, optional
        _description_, by default lora_alpha
    lora_dropout : float, optional
        standard dropout, by default lora_dropout
    r : int, optional
        rank of the low rank matrices, by default r
    bias : str, optional
        _description_, by default bias
    task_type : str, optional
        type of task that we will be fine-tuning for, by default task_type

    Returns
    -------
    LoraConfig
        _description_
    """
    return LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=r,
        bias=bias,
        task_type=task_type,
        target_modules=get_target_modules(model),
    )
