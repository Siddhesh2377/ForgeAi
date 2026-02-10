use super::config::{TrainingConfig, TrainingMethod, DatasetFormat};

/// Generate a Python training script for SFT / LoRA / QLoRA / Full fine-tune.
pub fn generate_sft_script(config: &TrainingConfig) -> String {
    let model_path = &config.model_path;
    let dataset_path = &config.dataset_path;
    let output_path = &config.output_path;

    let dataset_load = dataset_load_code(&config.dataset_format, dataset_path);
    let model_load = model_load_code(config);
    let lora_setup = lora_setup_code(config);
    let trainer_setup = sft_trainer_code(config);
    let merge_code = merge_adapter_code(config);

    format!(r#"#!/usr/bin/env python3
"""ForgeAI Training Script — Auto-generated"""
import json, sys, os, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from datasets import load_dataset
{peft_imports}
{trl_imports}

# ── Progress callback ──
class ForgeProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        elapsed = time.time() - self.start_time
        eta = None
        if state.global_step > 0 and state.max_steps > 0:
            rate = elapsed / state.global_step
            remaining = state.max_steps - state.global_step
            eta = int(rate * remaining)
        gpu_mem = None
        if torch.cuda.is_available():
            gpu_mem = int(torch.cuda.max_memory_allocated() / 1048576)
        progress = {{
            "type": "progress",
            "step": state.global_step,
            "total_steps": state.max_steps,
            "epoch": round(state.epoch, 2) if state.epoch else None,
            "loss": logs.get("loss"),
            "lr": logs.get("learning_rate"),
            "eta": eta,
            "gpu_mem": gpu_mem,
            "percent": round(state.global_step / max(state.max_steps, 1) * 100, 1),
        }}
        print(json.dumps(progress), flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        print(json.dumps({{"type": "status", "stage": "training", "message": "Training started"}}), flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        print(json.dumps({{"type": "status", "stage": "complete", "message": "Training complete"}}), flush=True)

# ── Load dataset ──
print(json.dumps({{"type": "status", "stage": "loading", "message": "Loading dataset..."}}), flush=True)
{dataset_load}

# ── Load model ──
print(json.dumps({{"type": "status", "stage": "loading", "message": "Loading model..."}}), flush=True)
{model_load}

tokenizer = AutoTokenizer.from_pretrained("{model_path}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

{lora_setup}

# ── Training ──
print(json.dumps({{"type": "status", "stage": "training", "message": "Starting training..."}}), flush=True)
{trainer_setup}

result = trainer.train()

# ── Save ──
print(json.dumps({{"type": "status", "stage": "saving", "message": "Saving model..."}}), flush=True)
trainer.save_model("{output_path}")
tokenizer.save_pretrained("{output_path}")

{merge_code}

# ── Report ──
import pathlib
total_size = sum(f.stat().st_size for f in pathlib.Path("{output_path}").rglob("*") if f.is_file())
print(json.dumps({{
    "type": "result",
    "output_path": "{output_path}",
    "output_size": total_size,
    "final_loss": result.training_loss if hasattr(result, "training_loss") else None,
    "epochs_completed": {epochs},
}}), flush=True)
"#,
        peft_imports = peft_imports(config),
        trl_imports = trl_imports(config),
        dataset_load = dataset_load,
        model_load = model_load,
        model_path = model_path,
        lora_setup = lora_setup,
        trainer_setup = trainer_setup,
        output_path = output_path,
        merge_code = merge_code,
        epochs = config.epochs,
    )
}

/// Generate a DPO training script.
pub fn generate_dpo_script(config: &TrainingConfig) -> String {
    let model_path = &config.model_path;
    let dataset_path = &config.dataset_path;
    let output_path = &config.output_path;
    let beta = config.dpo_beta.unwrap_or(0.1);
    let dataset_load = dataset_load_code(&config.dataset_format, dataset_path);
    let model_load = model_load_code(config);
    let lora_setup = lora_setup_code(config);
    let merge_code = merge_adapter_code(config);

    format!(r#"#!/usr/bin/env python3
"""ForgeAI DPO Training Script — Auto-generated"""
import json, sys, os, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, TrainerCallback
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
{peft_imports}

class ForgeProgressCallback(TrainerCallback):
    def __init__(self):
        self.start_time = time.time()
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        elapsed = time.time() - self.start_time
        eta = None
        if state.global_step > 0 and state.max_steps > 0:
            eta = int((elapsed / state.global_step) * (state.max_steps - state.global_step))
        gpu_mem = int(torch.cuda.max_memory_allocated() / 1048576) if torch.cuda.is_available() else None
        print(json.dumps({{
            "type": "progress",
            "step": state.global_step,
            "total_steps": state.max_steps,
            "epoch": round(state.epoch, 2) if state.epoch else None,
            "loss": logs.get("loss"),
            "lr": logs.get("learning_rate"),
            "eta": eta,
            "gpu_mem": gpu_mem,
            "percent": round(state.global_step / max(state.max_steps, 1) * 100, 1),
        }}), flush=True)

print(json.dumps({{"type": "status", "stage": "loading", "message": "Loading dataset..."}}), flush=True)
{dataset_load}

print(json.dumps({{"type": "status", "stage": "loading", "message": "Loading model..."}}), flush=True)
{model_load}

tokenizer = AutoTokenizer.from_pretrained("{model_path}", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

{lora_setup}

dpo_config = DPOConfig(
    output_dir="{output_path}",
    num_train_epochs={epochs},
    per_device_train_batch_size={batch_size},
    gradient_accumulation_steps={grad_accum},
    learning_rate={lr},
    warmup_steps={warmup},
    weight_decay={wd},
    beta={beta},
    logging_steps=1,
    save_strategy="steps",
    save_steps={save_steps},
    bf16=torch.cuda.is_available(),
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[ForgeProgressCallback()],
)

print(json.dumps({{"type": "status", "stage": "training", "message": "Starting DPO training..."}}), flush=True)
result = trainer.train()

print(json.dumps({{"type": "status", "stage": "saving", "message": "Saving model..."}}), flush=True)
trainer.save_model("{output_path}")
tokenizer.save_pretrained("{output_path}")

{merge_code}

import pathlib
total_size = sum(f.stat().st_size for f in pathlib.Path("{output_path}").rglob("*") if f.is_file())
print(json.dumps({{
    "type": "result",
    "output_path": "{output_path}",
    "output_size": total_size,
    "final_loss": result.training_loss if hasattr(result, "training_loss") else None,
    "epochs_completed": {epochs},
}}), flush=True)
"#,
        peft_imports = peft_imports(config),
        dataset_load = dataset_load,
        model_load = model_load,
        model_path = model_path,
        lora_setup = lora_setup,
        output_path = output_path,
        merge_code = merge_code,
        epochs = config.epochs,
        batch_size = config.batch_size,
        grad_accum = config.gradient_accumulation_steps,
        lr = config.learning_rate,
        warmup = config.warmup_steps,
        wd = config.weight_decay,
        beta = beta,
        save_steps = config.save_steps,
    )
}

// ── Helpers ─────────────────────────────────────────

fn peft_imports(config: &TrainingConfig) -> &'static str {
    match config.method {
        TrainingMethod::Lora | TrainingMethod::Qlora => {
            "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType"
        }
        _ => "# No PEFT imports needed for this method",
    }
}

fn trl_imports(config: &TrainingConfig) -> &'static str {
    match config.method {
        TrainingMethod::Sft | TrainingMethod::Lora | TrainingMethod::Qlora => {
            "from trl import SFTTrainer, SFTConfig"
        }
        TrainingMethod::Dpo => "from trl import DPOConfig, DPOTrainer",
        TrainingMethod::FullFinetune => "from transformers import Trainer",
    }
}

fn dataset_load_code(format: &DatasetFormat, path: &str) -> String {
    match format {
        DatasetFormat::Json => format!(r#"dataset = load_dataset("json", data_files="{}", split="train")"#, path),
        DatasetFormat::Jsonl => format!(r#"dataset = load_dataset("json", data_files="{}", split="train")"#, path),
        DatasetFormat::Csv => format!(r#"dataset = load_dataset("csv", data_files="{}", split="train")"#, path),
        DatasetFormat::Parquet => format!(r#"dataset = load_dataset("parquet", data_files="{}", split="train")"#, path),
    }
}

fn model_load_code(config: &TrainingConfig) -> String {
    match config.method {
        TrainingMethod::Qlora => {
            let bits = config.quantization_bits.unwrap_or(4);
            format!(
                r#"from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_{}bit=True,
    bnb_{}bit_compute_dtype=torch.bfloat16,
    bnb_{}bit_use_double_quant=True,
    bnb_{}bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "{}",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)"#,
                bits, bits, bits, bits, config.model_path
            )
        }
        TrainingMethod::FullFinetune => {
            format!(
                r#"model = AutoModelForCausalLM.from_pretrained(
    "{}",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)"#,
                config.model_path
            )
        }
        _ => {
            format!(
                r#"model = AutoModelForCausalLM.from_pretrained(
    "{}",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)"#,
                config.model_path
            )
        }
    }
}

fn lora_setup_code(config: &TrainingConfig) -> String {
    match config.method {
        TrainingMethod::Lora | TrainingMethod::Qlora => {
            let rank = config.lora_rank.unwrap_or(16);
            let alpha = config.lora_alpha.unwrap_or(rank * 2);
            let dropout = config.lora_dropout.unwrap_or(0.05);

            let target_modules = if let Some(ref modules) = config.target_modules {
                format!("[{}]", modules.iter().map(|m| format!("\"{}\"", m)).collect::<Vec<_>>().join(", "))
            } else {
                r#"["q_proj", "v_proj"]"#.to_string()
            };

            let layers_to_transform = if let Some(ref layers) = config.layers_to_transform {
                if layers.is_empty() {
                    "None".to_string()
                } else {
                    format!("[{}]", layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(", "))
                }
            } else {
                "None".to_string()
            };

            format!(
                r#"lora_config = LoraConfig(
    r={rank},
    lora_alpha={alpha},
    lora_dropout={dropout},
    target_modules={target_modules},
    layers_to_transform={layers_to_transform},
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()"#,
                rank = rank,
                alpha = alpha,
                dropout = dropout,
                target_modules = target_modules,
                layers_to_transform = layers_to_transform,
            )
        }
        _ => "# No LoRA config for this method".to_string(),
    }
}

fn sft_trainer_code(config: &TrainingConfig) -> String {
    match config.method {
        TrainingMethod::Sft | TrainingMethod::Lora | TrainingMethod::Qlora => {
            format!(
                r#"sft_config = SFTConfig(
    output_dir="{output}",
    num_train_epochs={epochs},
    per_device_train_batch_size={batch},
    gradient_accumulation_steps={grad_accum},
    learning_rate={lr},
    warmup_steps={warmup},
    weight_decay={wd},
    max_length={seq_len},
    logging_steps=1,
    save_strategy="steps",
    save_steps={save_steps},
    bf16=torch.cuda.is_available(),
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={{"use_reentrant": False}},
    optim="adamw_8bit",
    dataloader_pin_memory=False,
    max_grad_norm=0.3,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    callbacks=[ForgeProgressCallback()],
)"#,
                output = config.output_path,
                epochs = config.epochs,
                batch = config.batch_size,
                grad_accum = config.gradient_accumulation_steps,
                lr = config.learning_rate,
                warmup = config.warmup_steps,
                wd = config.weight_decay,
                seq_len = config.max_seq_length,
                save_steps = config.save_steps,
            )
        }
        TrainingMethod::FullFinetune => {
            format!(
                r#"training_args = TrainingArguments(
    output_dir="{output}",
    num_train_epochs={epochs},
    per_device_train_batch_size={batch},
    gradient_accumulation_steps={grad_accum},
    learning_rate={lr},
    warmup_steps={warmup},
    weight_decay={wd},
    logging_steps=1,
    save_strategy="steps",
    save_steps={save_steps},
    bf16=torch.cuda.is_available(),
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={{"use_reentrant": False}},
    optim="adamw_8bit",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[ForgeProgressCallback()],
)"#,
                output = config.output_path,
                epochs = config.epochs,
                batch = config.batch_size,
                grad_accum = config.gradient_accumulation_steps,
                lr = config.learning_rate,
                warmup = config.warmup_steps,
                wd = config.weight_decay,
                save_steps = config.save_steps,
            )
        }
        TrainingMethod::Dpo => "# DPO handled in separate script".to_string(),
    }
}

fn merge_adapter_code(config: &TrainingConfig) -> String {
    if !config.merge_adapter {
        return "# Adapter not merged (saved separately)".to_string();
    }
    match config.method {
        TrainingMethod::Lora | TrainingMethod::Qlora => {
            format!(
                r#"print(json.dumps({{"type": "status", "stage": "merging", "message": "Merging adapter into base model..."}}), flush=True)
merged_model = model.merge_and_unload()
merged_model.save_pretrained("{}")
print(json.dumps({{"type": "status", "stage": "merging", "message": "Adapter merged successfully"}}), flush=True)"#,
                config.output_path
            )
        }
        _ => "# Full model already saved, no adapter merge needed".to_string(),
    }
}
