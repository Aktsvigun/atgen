import torch

if torch.cuda.is_available():
    from unsloth import FastLanguageModel
else:

    class FastLanguageModel:
        pass


from bitsandbytes.nn import Linear4bit
from omegaconf import DictConfig
from torch.nn import Linear
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
    PeftMixedModel,
)


def prepare_model_for_training(
    model: FastLanguageModel, peft_config: DictConfig
) -> FastLanguageModel:
    if peft_config.use:
        target_modules = getattr(peft_config, "modules", None)
        if target_modules is None:
            target_modules = _find_all_linear_names(model)

        if torch.cuda.is_available():
            model = FastLanguageModel.get_peft_model(
                model,
                r=peft_config.r,
                target_modules=target_modules,
                lora_alpha=peft_config.lora_alpha,
                lora_dropout=peft_config.lora_dropout,
                bias=peft_config.bias,
                use_gradient_checkpointing=peft_config.use_gradient_checkpointing,
                random_state=peft_config.seed,
                use_rslora=False,
                loftq_config=None,
            )
        else:
            model = prepare_model_for_kbit_training(model)
            # Create PEFT config for these modules and wrap the model to PEFT
            peft_config = LoraConfig(
                r=peft_config.r,  # dimension of the updated matrices
                lora_alpha=peft_config.lora_alpha,  # parameter for scaling
                # TODO: make universal
                target_modules=target_modules,
                lora_dropout=peft_config.lora_dropout,  # dropout probability for layers
                bias=peft_config.bias,
                # TODO: maybe change for seq2seq model
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            # Ensure gradients are enabled for CPU training
            if not torch.cuda.is_available():
                for param in model.parameters():
                    param.requires_grad = True

    # Disable caching during training
    model.config.use_cache = False
    return model


def _find_all_linear_names(model):
    clss = [Linear4bit, Linear]
    lora_module_names = set()
    for name, module in model.named_modules():
        if any(isinstance(module, cls) for cls in clss):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
