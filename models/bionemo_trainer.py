"""
BioNeMo Framework: Fine-tune protein language models on custom datasets.
Uses NVIDIA BioNeMo + HuggingFace Transformers + DeepSpeed for distributed training.
SDKs: BioNeMo, Transformers, DeepSpeed, Accelerate, W&B
"""
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    EsmForMaskedLM, EsmTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
)
from accelerate import Accelerator
import wandb

try:
    import bionemo
    BIONEMO_AVAILABLE = True
except ImportError:
    BIONEMO_AVAILABLE = False


@dataclass
class BioNeMoTrainConfig:
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    output_dir: str = "./outputs/bionemo_finetune"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    fp16: bool = True
    gradient_accumulation_steps: int = 4
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    mlm_probability: float = 0.15
    max_length: int = 512
    wandb_project: str = "bio-compute-stack"
    deepspeed_config: Optional[str] = None


class ProteinSequenceDataset(Dataset):
    """Dataset of protein sequences for masked language modeling."""

    def __init__(self, sequences: List[str], tokenizer, max_length: int = 512):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].upper()
        encoding = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


class BioNeMoFineTuner:
    """
    Fine-tune ESM protein language models on custom sequence datasets.
    Supports masked language modeling, distributed training via DeepSpeed.
    """

    def __init__(self, config: BioNeMoTrainConfig = None):
        self.cfg = config or BioNeMoTrainConfig()
        print(f"[BioNeMo] Loading tokenizer: {self.cfg.model_name}")
        self.tokenizer = EsmTokenizer.from_pretrained(self.cfg.model_name)
        self.model = EsmForMaskedLM.from_pretrained(self.cfg.model_name)
        print(f"[BioNeMo] Model params: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_sequences_from_fasta(self, fasta_path: str) -> List[str]:
        """Parse sequences from FASTA file."""
        sequences = []
        current_seq = []
        with open(fasta_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
        print(f"[BioNeMo] Loaded {len(sequences)} sequences from {fasta_path}")
        return sequences

    def train(
        self,
        train_sequences: List[str],
        eval_sequences: Optional[List[str]] = None,
        run_name: Optional[str] = None,
    ):
        run_name = run_name or f"esm-finetune-{len(train_sequences)}seq"

        wandb.init(
            project=self.cfg.wandb_project,
            name=run_name,
            config=vars(self.cfg),
        )

        train_dataset = ProteinSequenceDataset(
            train_sequences, self.tokenizer, self.cfg.max_length
        )
        eval_dataset = ProteinSequenceDataset(
            eval_sequences or train_sequences[:100], self.tokenizer, self.cfg.max_length
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.cfg.mlm_probability,
        )

        training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.num_train_epochs,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.per_device_eval_batch_size,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            weight_decay=self.cfg.weight_decay,
            fp16=self.cfg.fp16,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            save_steps=self.cfg.save_steps,
            eval_steps=self.cfg.eval_steps,
            logging_steps=self.cfg.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            report_to=["wandb"],
            run_name=run_name,
            deepspeed=self.cfg.deepspeed_config,
            dataloader_num_workers=4,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        print(f"[BioNeMo] Starting training: {len(train_sequences)} seqs, {self.cfg.num_train_epochs} epochs")
        trainer.train()
        trainer.save_model(self.cfg.output_dir)
        self.tokenizer.save_pretrained(self.cfg.output_dir)
        print(f"[BioNeMo] Model saved to {self.cfg.output_dir}")
        wandb.finish()
        return trainer
