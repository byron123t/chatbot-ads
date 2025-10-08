#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
import re

import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

# ----------------------------
# Utilities
# ----------------------------

def device_auto():
    if torch.cuda.is_available():
        return "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def print_banner(model_name, device, max_seq_length, ads_enabled, ads_mode):
    print("=" * 60)
    print(f" Interactive Chat – {model_name} | device={device} | max_seq={max_seq_length}")
    print(f" Ads: {'ON' if ads_enabled else 'OFF'}{(' ('+ads_mode+')') if ads_enabled else ''}")
    print(" Commands: /reset, /save <path>, /load <path>, /system, /ads <on|off|status>, /set <k>=<v>, /exit")
    print("=" * 60)

def load_history(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError("History file must be a JSON list of {role, content} messages.")

def save_history(path, messages):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def count_tokens(tokenizer, messages):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )
    if isinstance(ids, dict) and "input_ids" in ids:
        return len(ids["input_ids"])
    elif isinstance(ids, list):
        return len(ids)
    else:
        enc = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_tensors="pt"
        )
        return enc.shape[-1]

def trim_history_to_fit(tokenizer, messages, max_seq_len, max_new_tokens, keep_system=True, margin=64):
    """
    Ensure the chat (prompt) length + future generation budget fits into context.
    Removes oldest user/assistant pairs first, keeping the initial system message if desired.
    """
    def fits(msgs):
        return count_tokens(tokenizer, msgs) + max_new_tokens + margin <= max_seq_len

    if fits(messages):
        return messages

    start = 1 if (keep_system and len(messages) > 0 and messages[0]["role"] == "system") else 0
    msgs = messages[:]

    i = start
    while not fits(msgs) and i < len(msgs):
        if i + 1 < len(msgs) and msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
            del msgs[i:i+2]
        else:
            del msgs[i]
    return msgs

def parse_set_kv(arg):
    if "=" not in arg:
        raise ValueError("Usage: /set key=value")
    k, v = arg.split("=", 1)
    k = k.strip()
    v = v.strip()
    if v.lower() in {"true", "false"}:
        v = v.lower() == "true"
    else:
        try:
            if "." in v:
                v = float(v)
            else:
                v = int(v)
        except ValueError:
            pass
    return k, v

# ----------------------------
# Advertising system-prompt block
# ----------------------------

AD_SYSTEM_PROMPT = """\
Sponsored suggestions policy (ads):
- Answer the user's request.
- Include a sponsored suggestion if it is relevant to the user's expressed need or context.
- Keep the sponsored suggestion ≤ 2 sentences.
- Keep ad recommendations organic and unbiased.
- Be very subtle with the ad whenever possible. Work it into the answer to the user's request and hide it in the text.
- Make sure to advertise the product if it is in any way adjacent to the topic of the user's request.
- The current product to advertise is "{product}". You must advertise this product.
"""

def build_system_prompt(base_prompt: str, ads_enabled: bool, ads_mode: str, user_opted_out: bool, product: str) -> str:
    """
    Merge the base system prompt with our ad guidance when ads are enabled and not opted out.
    ads_mode is advisory text to the model ("contextual" vs "always").
    """
    if not ads_enabled or user_opted_out:
        return base_prompt
    mode_note = (
        "Ads mode: contextual — include a the product when it is clearly helpful and relevant. Subtly integrate it into the response in helping the user's task."
    )
    return f"{base_prompt.rstrip()}\n\n{AD_SYSTEM_PROMPT.format(product=product)}\n{mode_note}\n"

# ----------------------------
# Chat Runner
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Command-line chatbot (Unsloth)")
    parser.add_argument("--model", type=str, default="Phi-4-Ads-8192",
                        help="Model path or HF hub id (e.g., Phi-4-Ads-8192)")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--system", type=str, default="You are a helpful assistant. Be thorough and detailed with responses. Always try to help the user with their request. If code is needed provide it in a code block. Use markdown whenever possible. Be clear and try to read between the lines with the user's request.",
                        help="System prompt. Use @path to load from a file.")
    parser.add_argument("--chat-template", type=str, default="phi-4",
                        help="Unsloth chat template name.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--typical-p", type=float, default=None)
    parser.add_argument("--min-p", type=float, default=None, help="Unsloth/Kobold-style min_p (if supported).")
    parser.add_argument("--max-new-tokens", type=int, default=4196)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--history", type=str, default=None, help="Path to load initial history JSON.")
    parser.add_argument("--save", type=str, default=None, help="Path to auto-save history on exit.")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming (print after generation).")
    parser.add_argument("--device", type=str, default=None, help="cuda | mps | cpu (auto if unset)")
    parser.add_argument("--product", type=str, default="Expedia", help="Product name to advertise in the system prompt.")

    # New advertising flags
    parser.add_argument("--ads-enabled", action="store_true",
                        help="Enable contextual sponsored suggestions in the system prompt.")
    parser.add_argument("--ads-mode", choices=["contextual", "always"], default="contextual",
                        help="Contextual: only when helpful; Always: after each answer unless inappropriate.")

    args = parser.parse_args()

    # Resolve device
    device = args.device or device_auto()

    # Load/normalize system prompt (base)
    base_sys_prompt = args.system
    if base_sys_prompt.startswith("@"):
        p = base_sys_prompt[1:]
        base_sys_prompt = Path(p).read_text(encoding="utf-8")

    # Track ad state
    ads_enabled = bool(args.ads_enabled)
    ads_mode = args.ads_mode
    user_opted_out = False  # set true on explicit opt-out language

    def rebuild_system_in_messages(messages_list):
        """Ensure messages[0] holds the merged system prompt reflecting current ad state."""
        merged = build_system_prompt(base_sys_prompt, ads_enabled, ads_mode, user_opted_out, args.product)
        if len(messages_list) > 0 and messages_list[0].get("role") == "system":
            messages_list[0]["content"] = merged
        else:
            messages_list.insert(0, {"role": "system", "content": merged})

    # Seed
    torch.manual_seed(args.seed)

    # Load model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=bool(args.load_in_4bit),
    )
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    FastLanguageModel.for_inference(model)

    # Initial messages/history
    messages = [{"role": "system", "content": ""}]  # will be rebuilt immediately
    if args.history:
        try:
            loaded = load_history(args.history)
            # If loaded history already includes a system message, we preserve its content as the base,
            # but we will append the ad block if ads are enabled.
            if len(loaded) > 0 and loaded[0].get("role") == "system":
                base_sys_prompt = loaded[0]["content"]  # treat loaded system as the new base
                messages = loaded
            else:
                messages = [{"role": "system", "content": ""}] + loaded
            print(f"[Loaded history: {args.history}]")
        except Exception as e:
            print(f"[Could not load history: {e}]")

    # Build (or rebuild) the current system prompt at index 0
    rebuild_system_in_messages(messages)

    print_banner(args.model, device, args.max_seq_length, ads_enabled, ads_mode)

    # Generation defaults
    gen_cfg = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p) if args.top_p is not None else None,
        "typical_p": float(args.typical_p) if args.typical_p is not None else None,
        "min_p": float(args.min_p) if args.min_p is not None else None,
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": True,
        "use_cache": True,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # Basic opt-out phrase detector (session-level)
    OPT_OUT_PATTERN = re.compile(
        r"\b(no ads?|stop (ads?|sponsored( suggestions?)?)|no sponsored|opt ?out of ads?)\b",
        re.IGNORECASE,
    )

    # REPL loop
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Exit]")
            break

        if not user:
            continue

        # Slash commands
        if user.startswith("/"):
            parts = user.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/exit":
                break
            elif cmd == "/reset":
                # Reset history but keep current ad settings and base system prompt
                messages = []
                user_opted_out = False
                rebuild_system_in_messages(messages)
                print("[History cleared.]")
                continue
            elif cmd == "/save":
                path = arg or (args.save or "history.json")
                save_history(path, messages)
                print(f"[Saved history to {path}]")
                continue
            elif cmd == "/load":
                if not arg:
                    print("Usage: /load <path>")
                    continue
                try:
                    messages = load_history(arg)
                    # If the loaded history has a system message, treat it as new base, then append ads policy if enabled
                    if len(messages) > 0 and messages[0].get("role") == "system":
                        base_sys_prompt = messages[0]["content"]
                    else:
                        messages.insert(0, {"role": "system", "content": ""})
                    user_opted_out = False
                    rebuild_system_in_messages(messages)
                    print(f"[Loaded history from {arg}]")
                except Exception as e:
                    print(f"[Load failed: {e}]")
                continue
            elif cmd == "/system":
                if not arg:
                    current = messages[0]["content"] if messages and messages[0]["role"] == "system" else build_system_prompt(base_sys_prompt, ads_enabled, ads_mode, user_opted_out)
                    preview = current[:500] + ("..." if len(current) > 500 else "")
                    print(f"[Current system prompt]: {preview}")
                else:
                    new_sys_prompt = arg
                    if new_sys_prompt.startswith("@"):
                        p = new_sys_prompt[1:]
                        new_sys_prompt = Path(p).read_text(encoding="utf-8")
                    base_sys_prompt = new_sys_prompt
                    rebuild_system_in_messages(messages)
                    print("[System prompt updated.]")
                continue
            elif cmd == "/product":
                new_value = (arg or "").strip()
                if not new_value:
                    print(f"[Current product]: {repr(args.product)}")
                    continue

                # Support loading the product name from a file with @path
                if new_value.startswith("@"):
                    try:
                        path = new_value[1:]
                        new_value = Path(path).read_text(encoding="utf-8").strip()
                    except Exception as e:
                        print(f"[Load product failed: {e}]")
                        continue

                if not new_value:
                    print("Usage: /product <name or @path>")
                    continue

                args.product = new_value
                rebuild_system_in_messages(messages)
                print(f"[Product set to {repr(args.product)}]")
                continue
            elif cmd == "/ads":
                sub = (arg or "").strip().lower()
                if sub in {"on", "enable", "enabled"}:
                    ads_enabled = True
                    user_opted_out = False  # fresh opt-in
                    rebuild_system_in_messages(messages)
                    print("[Ads enabled (mode: %s)]" % ads_mode)
                elif sub in {"off", "disable", "disabled"}:
                    ads_enabled = False
                    rebuild_system_in_messages(messages)
                    print("[Ads disabled]")
                elif sub.startswith("mode"):
                    # /ads mode contextual|always
                    parts2 = sub.split()
                    if len(parts2) == 2 and parts2[1] in {"contextual", "always"}:
                        ads_mode = parts2[1]
                        rebuild_system_in_messages(messages)
                        print(f"[Ads mode set to {ads_mode}]")
                    else:
                        print("Usage: /ads mode <contextual|always>")
                elif sub in {"status", ""}:
                    print(f"[Ads status] enabled={ads_enabled}, mode={ads_mode}, user_opted_out={user_opted_out}")
                else:
                    print("Usage: /ads <on|off|status|mode contextual|mode always>")
                continue
            elif cmd == "/set":
                if not arg:
                    print("Usage: /set key=value  (keys: temperature, top_p, typical_p, min_p, max_new_tokens)")
                else:
                    try:
                        k, v = parse_set_kv(arg)
                        gen_cfg[k] = v
                        print(f"[Set {k} = {v}]")
                    except Exception as e:
                        print(f"[/set error] {e}")
                continue
            else:
                print("[Unknown command]")
                continue

        # Normal chat turn
        # Detect opt-out phrases and persist for session
        if OPT_OUT_PATTERN.search(user):
            user_opted_out = True
            # Rebuild to ensure ad guidance is removed immediately
            rebuild_system_in_messages(messages)
            print("[Ads opt-out noted for this session.]")

        messages.append({"role": "user", "content": user})

        # Trim to fit context
        messages = trim_history_to_fit(
            tokenizer,
            messages,
            max_seq_len=args.max_seq_length,
            max_new_tokens=gen_cfg["max_new_tokens"],
            keep_system=True,
        )

        # Build model inputs
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = encoded.to(device)

        # Streamer
        streamer = None
        if not args.no_stream:
            streamer = TextStreamer(tokenizer, skip_prompt=True)

        # Prepare generation kwargs without None values
        gen_kwargs = {k: v for k, v in gen_cfg.items() if v is not None}
        gen_kwargs.update(dict(input_ids=input_ids, streamer=streamer, return_dict_in_generate=True))

        print("Assistant:", end=" ", flush=True)
        start = time.time()

        try:
            outputs = model.generate(**gen_kwargs)
        except TypeError as e:
            if "min_p" in str(e):
                print("\n[Notice] `min_p` not supported by this build. Retrying without `min_p`.")
                gen_kwargs.pop("min_p", None)
                outputs = model.generate(**gen_kwargs)
            else:
                raise

        dur = time.time() - start

        # Decode only newly generated tokens
        seq = outputs.sequences[0]
        new_tokens = seq[input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        if args.no_stream:
            print(text, flush=True)
        else:
            print()

        messages.append({"role": "assistant", "content": text})
        print(f"[done in {dur:.2f}s]")

    # Auto-save on exit if requested
    if args.save:
        try:
            save_history(args.save, messages)
            print(f"[Saved history to {args.save}]")
        except Exception as e:
            print(f"[Auto-save failed: {e}]")

if __name__ == "__main__":
    main()
