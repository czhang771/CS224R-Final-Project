"""Starter SFT training entrypoint for the class project.

This file is intentionally incomplete. Students are expected to implement
`train(...)` while reusing the data/model setup provided here.
"""

import sys
from pathlib import Path

# Allow `python sft_trainer/sft.py` to resolve imports from project root.
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
import gc
import argparse
import os
from sft_trainer.sft_dataset import get_dataloaders
import wandb
import torch.nn.functional as F
import tqdm.auto as tqdm
# os.environ['WANDB_MODE'] = 'offline'

def get_model(model_name, device='cuda', use_gradient_checkpointing=True):
    """Load policy model + tokenizer for SFT training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory (trades compute for memory)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    model.train()
    return model, tokenizer

def clear_cache(model):
    """Best-effort GPU/CPU cache cleanup between heavy steps."""
    torch.cuda.empty_cache()
    gc.collect()

def save_checkpoint(model, tokenizer, optimizer, scheduler, output_dir):
    """Save model/tokenizer plus optimizer/scheduler states."""
    os.makedirs(output_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model and tokenizer saved to {model_dir}")

    torch.save({
        'scheduler': scheduler.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(output_dir, 'train_states.pth'))
    print(f"Model saved to {output_dir}")

def train(
    model, 
    tokenizer, 
    train_dataloader, 
    test_dataloader, 
    optimizer, 
    scheduler, 
    num_epochs, 
    device='cuda', 
    save_model=1, 
    output_dir='sft_model', 
    gradient_accumulation_steps=1, 
    gradient_clipping=1.0
):
    # TODO(student): implement the SFT optimization loop.
    # Expected high-level flow:
    # 1) Forward pass on `input_ids` and compute token-level log-probs.
    # 2) Mask loss to response tokens only using `is_response_token`.
    # 3) Backprop, optionally clip gradients, then optimizer/scheduler steps.
    # 4) Periodically evaluate on `test_dataloader` and log metrics to W&B.
    # 5) Save checkpoints under `output_dir` when requested.

    model.train()
    global_step = 0
    optimizer.zero_grad()
    model_device = next(model.parameters()).device

    def compute_loss_and_accuracy(batch):
        input_ids = batch['input_ids'].to(model_device)
        attention_mask = batch['attention_mask'].to(model_device)
        is_response_token = batch['is_response_token'].to(model_device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_response_mask = is_response_token[:, 1:] * attention_mask[:, 1:]
        
        loss_per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        )

        loss_per_token = loss_per_token.view(shift_labels.shape)
        masked_loss = loss_per_token * shift_response_mask

        num_response_tokens = shift_response_mask.sum().clamp(min=1)
        loss = masked_loss.sum() / num_response_tokens

        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels) * shift_response_mask
        token_accuracy = correct.sum() / num_response_tokens
        
        return loss, token_accuracy

    for epoch in range(num_epochs):
        model.train()

        progress_bar = tqdm.tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{num_epochs}"
        )

        running_loss = 0.0
        running_token_accuracy = 0.0

        for batch_idx, batch in progress_bar:
            loss, token_accuracy = compute_loss_and_accuracy(batch)

            raw_loss = loss.detach()
            raw_token_accuracy = token_accuracy.detach()

            loss = loss / gradient_accumulation_steps
            loss.backward()

            running_loss += raw_loss.item()
            running_token_accuracy += raw_token_accuracy.item()

            if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clipping
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                lr = scheduler.get_last_lr()[0]
                avg_train_loss = running_loss / (batch_idx + 1)
                avg_train_token_accuracy = running_token_accuracy / (batch_idx + 1)

                log_dict = {
                    "train/loss": raw_loss.item(),
                    "train/token_accuracy": raw_token_accuracy.item(),
                    "train/avg_loss_epoch_so_far": avg_train_loss,
                    "train/avg_token_accuracy_epoch_so_far": avg_train_token_accuracy,
                    "train/learning_rate": lr,
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                    "train/grad_norm": grad_norm.item(),
                }

                wandb.log(log_dict, step=global_step)

                progress_bar.set_postfix({
                    "loss": raw_loss.item(),
                    "acc": raw_token_accuracy.item(),
                    "avg_loss": avg_train_loss,
                    "avg_acc": avg_train_token_accuracy,
                    "lr": lr,
                })

        model.eval()
        eval_losses = []
        eval_token_accuracies = []

        with torch.no_grad():
            eval_progress_bar = tqdm.tqdm(
                test_dataloader,
                total=len(test_dataloader),
                desc=f"Evaluating epoch {epoch + 1}/{num_epochs}"
            )

            for batch in eval_progress_bar:
                eval_loss, eval_token_accuracy = compute_loss_and_accuracy(batch)

                eval_losses.append(eval_loss.item())
                eval_token_accuracies.append(eval_token_accuracy.item())

                eval_progress_bar.set_postfix({
                    "eval/loss": eval_loss.item(),
                    "eval/acc": eval_token_accuracy.item(),
                })

        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        avg_eval_token_accuracy = sum(eval_token_accuracies) / len(eval_token_accuracies)
        eval_ppl = torch.exp(torch.tensor(avg_eval_loss)).item()

        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/token_accuracy": avg_eval_token_accuracy,
                "eval/perplexity": eval_ppl,
                "eval/epoch": epoch + 1,
                "eval/global_step": global_step,
            },
            step=global_step
        )

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={running_loss / len(train_dataloader):.4f} | "
            f"train_acc={running_token_accuracy / len(train_dataloader):.4f} | "
            f"eval_loss={avg_eval_loss:.4f} | "
            f"eval_acc={avg_eval_token_accuracy:.4f} | "
            f"eval_ppl={eval_ppl:.4f}"
        )

        if save_model:
            epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch + 1}")
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                epoch_output_dir
            )

        clear_cache(model)

    if save_model:
        final_output_dir = os.path.join(output_dir, "final")
        save_checkpoint(
            model,
            tokenizer,
            optimizer,
            scheduler,
            final_output_dir
        )

    print("Training complete.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-0.5B')
    parser.add_argument('--dataset_name', type=str, default='Asap7772/cog_behav_all_strategies')
    parser.add_argument('--output_dir', type=str, default='sft_model')
    parser.add_argument('--max_prompt_length', type=int, default=512)
    parser.add_argument('--max_response_length', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--wandb_project', type=str, default='sft_default_project')
    parser.add_argument('--wandb_name', type=str, default='test')
    parser.add_argument('--save_model', type=int, default=1)
    parser.add_argument('--gradient_checkpointing', type=int, default=1)
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_name)
    wandb.config.update(vars(args))

    model, tokenizer = get_model(args.model_name, args.device, use_gradient_checkpointing=args.gradient_checkpointing)

    dataloaders = get_dataloaders(
        dataset_name=args.dataset_name, 
        tokenizer=tokenizer, 
        max_prompt_length=args.max_prompt_length, 
        max_response_length=args.max_response_length, 
        batch_size=args.batch_size, 
        splits=['train', 'test'],
        pin_memory=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    train_dataloader, test_dataloader = dataloaders['train'], dataloaders['test']
    # Scheduler steps happen only after an optimizer step, so account for
    # gradient accumulation when estimating total training steps.
    num_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    warmup_steps = int(num_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_steps)

    full_output_dir = os.path.join(args.output_dir, args.wandb_project, args.wandb_name)
    os.makedirs(full_output_dir, exist_ok=True)

    train(
        model, 
        tokenizer, 
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        scheduler, 
        args.num_epochs, 
        args.device, 
        args.save_model, 
        full_output_dir, 
        args.gradient_accumulation_steps, 
        args.gradient_clipping
    )

if __name__ == "__main__":
    main()
