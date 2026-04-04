"""
Evaluation framework for VIA-SD on QA and generation datasets.

Extends evaluation_llama/eval.py with:
  - WebQuestions, NaturalQuestions (NQ), TriviaQA  [Table 1 & 2]
  - XSum, WMT14 (en→de), CNN/DM                    [Figure 2]
  - Speedup (τ) and acceptance-rate (r) metrics     [matching paper §3]

Metrics (paper Appendix / eval.py convention):
  r  = (sum(accept_lengths) - len(accept_lengths)) / draft_token_num
         "net accepted tokens" / "total draft tokens"
  τ  = mean(tokens/sec of method) / mean(tokens/sec of vanilla AR baseline)
       computed AFTER the fact by speed.py comparing two jsonl files
"""

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


# ── reproducibility ────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── dataset loading ────────────────────────────────────────────────────────────

def load_qa_data(
    task_name: str,
    seed: int,
    data_num: int,
    local_dataset_root: str = "",
    local_dataset_map: dict | None = None,
):
    """
    Load a QA or generation dataset and return (records, prompt_prefix).

    Each record is a dict passed to build_prompt().
    Returns:
        data        : list of dataset records
        prompt_shots: few-shot prefix string (empty for QA tasks)
    """
    limit_desc = "all" if data_num is None or data_num <= 0 else str(data_num)
    logging.info(f"Loading dataset '{task_name}' (n={limit_desc}, seed={seed})")

    def _shuffle_and_limit(ds):
        ds = ds.shuffle(seed=seed)
        if data_num is not None and data_num > 0:
            ds = ds.select(range(min(data_num, len(ds))))
        return list(ds)

    local_dataset_map = local_dataset_map or {}
    local_path = local_dataset_map.get(task_name, "")
    if not local_path and local_dataset_root:
        local_path = str(Path(local_dataset_root) / task_name)
    if local_path:
        p = Path(local_path)
        if p.exists():
            parquet_files = sorted(str(x) for x in p.rglob("*.parquet"))
            if parquet_files:
                logging.info(f"Loading local parquet dataset for {task_name}: {local_path}")
                raw = load_dataset("parquet", data_files=parquet_files, split="train")
                return _shuffle_and_limit(raw), ""
            logging.warning(f"Local path exists but no parquet found: {local_path}")
        else:
            logging.warning(f"Local path not found, fallback to remote dataset: {local_path}")

    if task_name == "webquestions":
        raw = load_dataset("web_questions", split="test", trust_remote_code=True)
        data = _shuffle_and_limit(raw)
        return data, ""

    elif task_name == "nq":
        # natural_questions validation split (test labels are private)
        raw = load_dataset("natural_questions", split="validation",
                           trust_remote_code=True)
        data = _shuffle_and_limit(raw)
        return data, ""

    elif task_name == "triviaqa":
        raw = load_dataset("trivia_qa", "rc", split="validation",
                           trust_remote_code=True)
        data = _shuffle_and_limit(raw)
        return data, ""

    elif task_name == "cnndm":
        n_shot = 1
        raw = load_dataset("cnn_dailymail", "3.0.0", split="test")
        shots = load_dataset("cnn_dailymail", "3.0.0", split="train"
                             ).shuffle(seed=seed).select(range(n_shot))
        prompt_shots = ""
        for s in shots:
            prompt_shots += (
                "Article: " + s["article"] +
                "\nSummary: " + s["highlights"].replace("\n", "") + "\n"
            )
        return _shuffle_and_limit(raw), prompt_shots

    elif task_name == "xsum":
        raw = load_dataset("xsum", split="test")
        return _shuffle_and_limit(raw), ""

    elif task_name == "wmt14":
        raw = load_dataset("wmt14", "de-en", split="test")
        return _shuffle_and_limit(raw), ""

    else:
        raise ValueError(f"Unknown task: {task_name}")


def build_prompt(record: dict, task_name: str, prompt_shots: str = "") -> str:
    """Convert a dataset record to a prompt string."""

    if task_name == "webquestions":
        return f"Question: {record['question']}\nAnswer:"

    elif task_name == "nq":
        # NQ: question is in record['question']['text']
        q = record["question"]["text"] if isinstance(record["question"], dict) \
            else record["question"]
        return f"Question: {q}\nAnswer:"

    elif task_name == "triviaqa":
        return f"Question: {record['question']}\nAnswer:"

    elif task_name == "cnndm":
        return prompt_shots + "Article: " + record["article"] + "\nSummary:"

    elif task_name == "xsum":
        return (
            "Summarize the following document into one sentence.\n"
            "Document: " + record["document"] + "\nSummary:"
        )

    elif task_name == "wmt14":
        # German → English translation
        src = record["translation"]["de"]
        return f"Translate from German to English.\nGerman: {src}\nEnglish:"

    else:
        raise ValueError(f"Unknown task: {task_name}")


def clip_to_max_length(
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    max_total: int = 4096,
) -> torch.Tensor:
    """Tokenize and clip prompt so total length fits within max_total."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    max_prompt_len = max_total - max_new_tokens
    if ids.shape[1] > max_prompt_len:
        ids = ids[:, :max_prompt_len]
    return ids  # [1, L]


# ── core evaluation loop ───────────────────────────────────────────────────────

@torch.inference_mode()
def get_model_answers_qa(
    model,
    tokenizer,
    forward_func,
    model_id: str,
    data: list,
    prompt_shots: str,
    answer_file: str,
    max_new_tokens: int,
    task_name: str,
    **kwargs,
):
    """
    Iterate over dataset records, call forward_func, write results to jsonl.

    forward_func signature:
        (input_ids, model, tokenizer, max_new_tokens, **kwargs)
        → (output_ids, new_token_num, step, accept_length_list, draft_token_num)
    """
    model.eval()
    os.makedirs(os.path.dirname(os.path.abspath(answer_file)), exist_ok=True)

    accept_lengths_all = []
    draft_token_num_all = 0

    for record in tqdm(data, desc=f"[{model_id}] {task_name}"):
        prompt = build_prompt(record, task_name, prompt_shots)
        max_total = getattr(model.config, "max_position_embeddings", 4096)
        max_total = min(max_total, 4096)
        input_ids = clip_to_max_length(tokenizer, prompt, max_new_tokens, max_total)
        input_ids = input_ids.to(next(model.parameters()).device)

        torch.cuda.synchronize()
        t0 = time.time()
        output_ids, new_token_num, step, accept_lengths, draft_token_num = forward_func(
            input_ids, model, tokenizer, max_new_tokens, **kwargs
        )
        torch.cuda.synchronize()
        wall_time = time.time() - t0

        # Decode output (strip prompt)
        gen_ids = output_ids[0][input_ids.shape[1]:]
        output_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # accepted_ratio = net accepted draft tokens / total draft tokens
        accepted_ratio = (
            (sum(accept_lengths) - len(accept_lengths)) / draft_token_num
            if draft_token_num > 0 else 0.0
        )
        rejection_rate = 1.0 - accepted_ratio

        accept_lengths_all.extend(accept_lengths)
        draft_token_num_all += draft_token_num

        entry = {
            "model_id": model_id,
            "task": task_name,
            "choices": [{
                "turns": output_text,
                "new_tokens": [int(new_token_num)],
                "wall_time": [wall_time],
                "decoding_steps": [int(step)],
                "accept_lengths": accept_lengths,
                "acceptance_rate": float(accepted_ratio),
                "rejection_rate": float(rejection_rate),
            }],
            "tstamp": time.time(),
        }
        with open(answer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    if accept_lengths_all:
        overall_accept_ratio = (
            (sum(accept_lengths_all) - len(accept_lengths_all)) / draft_token_num_all
            if draft_token_num_all > 0 else 0.0
        )
        overall_reject_rate = 1.0 - overall_accept_ratio
        summary = {
            "model_id": model_id,
            "task": task_name,
            "mean_accept_length": float(np.mean(accept_lengths_all)),
            "overall_acceptance_rate": float(overall_accept_ratio),
            "overall_rejection_rate_r": float(overall_reject_rate),
            "total_draft_tokens": int(draft_token_num_all),
            "n_samples": len(accept_lengths_all),
        }
        with open(answer_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        logging.info(
            f"[{model_id}] {task_name}  "
            f"r={overall_reject_rate:.3f}  "
            f"mean_accept={np.mean(accept_lengths_all):.2f}  "
            f"n={len(data)}"
        )


def run_eval_qa(
    model,
    tokenizer,
    forward_func,
    model_id: str,
    answer_file: str,
    max_new_tokens: int,
    task_name: str,
    data_num: int = -1,
    seed: int = 42,
    local_dataset_root: str = "",
    local_dataset_map: dict | None = None,
    **kwargs,
):
    """
    Top-level entry point — analogous to eval.py's run_eval().

    Usage:
        from evaluation_llama.eval_qa import run_eval_qa
        run_eval_qa(model, tokenizer, forward_fn, model_id="via-sd-3tier",
                    answer_file="test/...", max_new_tokens=512,
                    task_name="webquestions", data_num=200)
    """
    seed_everything(seed)
    data, prompt_shots = load_qa_data(
        task_name,
        seed,
        data_num,
        local_dataset_root=local_dataset_root,
        local_dataset_map=local_dataset_map,
    )

    get_model_answers_qa(
        model=model,
        tokenizer=tokenizer,
        forward_func=forward_func,
        model_id=model_id,
        data=data,
        prompt_shots=prompt_shots,
        answer_file=answer_file,
        max_new_tokens=max_new_tokens,
        task_name=task_name,
        **kwargs,
    )
