from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from fine_tuning_configs.general_config import base_model,save_dir

def merge_and_save(save_dir:str=save_dir,base_model:str=base_model):
    """
    Will merge the fine-tuned weights to the original model and save it to the chosen dir

    Parameters
    ----------
    save_dir : str
        fine-tuned model save directory
    
    base_model : str
        HF identifier of the base model

    Returns
    -------
    A saved & merged fine-tuned model in the selected local directory
    """
    # Reload model in FP16 and merge it with LoRA weights
    base_model = AutoModelForCausalLM.from_pretrained(
                base_model,
                low_cpu_mem_usage=True,
                return_dict=True,
                torch_dtype=torch.float16,
                device_map={"": 0},
            )
    try:
        model = PeftModel.from_pretrained(base_model, save_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(save_dir, safe_serialization=True)
        return "Merge done"
    except Exception as e:
        return f"An unexpected error as occured: {e}"