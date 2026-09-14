"""Reproducible LoRA SFT with answer-only masking and held-out validation."""
import argparse
import hashlib
import json
import math
import platform
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


def load_data(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or any(not row.get("question") or not row.get("answer") for row in records):
        raise ValueError("Dataset requires nonempty question and answer fields")
    return records


def encode_record(tokenizer, record, max_length=192):
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are an XR engineering assistant. Answer briefly using evidence."},
        {"role": "user", "content": record["question"]}], tokenize=True, add_generation_prompt=True)
    answer = tokenizer.encode(record["answer"], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Reserve response space explicitly; never train on prompt or padding tokens.
    answer = answer[:max_length // 2]
    prompt = prompt[-(max_length - len(answer)):]
    ids = prompt + answer
    return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "labels": torch.tensor([[-100]*len(prompt) + answer])}


def validation_loss(model, examples):
    model.eval()
    loss_sum, count = 0.0, 0
    with torch.inference_mode():
        for example in examples:
            batch = {key: value.to(model.device) for key, value in example.items()}
            n = int((batch["labels"][:, 1:] != -100).sum())
            loss_sum += float(model(**batch).loss) * n
            count += n
    return loss_sum / count


def tiny_base(train):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast
    engine = Tokenizer(WordLevel(unk_token="[UNK]"))
    engine.pre_tokenizer = Whitespace()
    engine.train_from_iterator([r["question"] + " " + r["answer"] for r in train],
                               WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]", "[BOS]"]))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=engine, unk_token="[UNK]", pad_token="[PAD]",
                                       eos_token="[EOS]", bos_token="[BOS]")
    tokenizer.chat_template = "{% for m in messages %}{{m['role']}}: {{m['content']}}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=512, bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return Qwen2ForCausalLM(config), tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/qwen2.5-0.5b-instruct")
    parser.add_argument("--tiny", action="store_true", help="Random small model; training pipeline smoke only")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    train_path, valid_path = root/"examples/sft_train.jsonl", root/"examples/sft_eval.jsonl"
    train, valid = load_data(train_path), load_data(valid_path)
    if {r["question"].strip().lower() for r in train} & {r["question"].strip().lower() for r in valid}:
        raise ValueError("Training and validation prompts overlap")
    started = time.perf_counter()
    kwargs = {"local_files_only": True, "torch_dtype": torch.float32}
    if args.qlora:
        if args.tiny or not torch.cuda.is_available():
            raise SystemExit("QLoRA requires a pretrained model, compatible CUDA and bitsandbytes")
        from transformers import BitsAndBytesConfig
        kwargs = {"local_files_only": True, "device_map": "auto", "quantization_config":
                  BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.float16)}
    if args.tiny:
        base, tokenizer = tiny_base(train)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        base = prepare_model_for_kbit_training(base)
    base.config.use_cache = False
    train_examples = [encode_record(tokenizer, r) for r in train]
    validation = [encode_record(tokenizer, r) for r in valid]
    before = validation_loss(base, validation)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                           target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    losses = []
    for step in range(args.steps):
        model.train()
        example = train_examples[step % len(train_examples)]
        batch = {key: value.to(model.device) for key, value in example.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
        optimizer.step()
        losses.append(round(float(loss.detach()), 6))
        print(json.dumps({"step": step+1, "loss": losses[-1]}), flush=True)
    after = validation_loss(model, validation)
    run_name = "tiny-lora" if args.tiny else ("qwen-qlora" if args.qlora else "qwen-lora")
    adapter_dir = root / "models" / run_name
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    sample = encode_record(tokenizer, valid[0])
    with torch.inference_mode():
        reference = model(input_ids=sample["input_ids"].to(model.device)).logits.detach().cpu()
    # Reload the saved adapter into the same frozen base; compare deterministic logits.
    model.load_adapter(str(adapter_dir), adapter_name="reloaded")
    model.set_adapter("reloaded")
    model.eval()
    with torch.inference_mode():
        reloaded = model(input_ids=sample["input_ids"].to(model.device)).logits.detach().cpu()
    delta = float((reference - reloaded).abs().max())
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": valid[0]["question"]}],
                                           tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=48,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    report = {"experiment": run_name, "base_model": "random-tiny-qwen2" if args.tiny else args.base_model,
              "pretrained_base": not args.tiny, "quantized": args.qlora, "seed": args.seed,
              "device": str(model.device), "platform": platform.system(), "torch": torch.__version__,
              "steps": args.steps, "train_examples": len(train), "validation_examples": len(valid),
              "train_data_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
              "eval_data_sha256": hashlib.sha256(valid_path.read_bytes()).hexdigest(),
              "answer_only_masking": True, "held_out_prompt_overlap": 0,
              "validation_loss_before": before, "validation_loss_after": after,
              "validation_perplexity_before": math.exp(min(before, 50)),
              "validation_perplexity_after": math.exp(min(after, 50)), "training_losses": losses,
              "adapter_reload_max_abs_delta": delta, "trainable_parameters": trainable_parameters,
              "total_parameters": total_parameters,
              "sample_question": valid[0]["question"],
              "sample_answer": tokenizer.decode(generated[0, prompt.shape[1]:], skip_special_tokens=True),
              "duration_seconds": time.perf_counter()-started,
              "scope": "Small original instructional dataset; workflow evidence, not a production capability benchmark."}
    out = root / "evidence" / (run_name + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(out), "validation_before": before, "validation_after": after, "reload_delta": delta}), flush=True)


if __name__ == "__main__":
    main()
