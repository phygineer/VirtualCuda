"""Chat with a model in the terminal.

Usage:
    .venv/bin/python chat.py                         # interactive chat with the default model
    .venv/bin/python chat.py --model <hf-repo-id>    # any standard causal LM
    .venv/bin/python chat.py --info                  # print architecture summary and exit

Commands inside chat: /reset clears history, /quit exits.
"""

import argparse
from threading import Thread

from transformers import TextIteratorStreamer

from models import DEFAULT_MODEL, LoadedModel


def print_info(lm):
    for k, v in lm.info().items():
        print(f"{k + ':':<20}{v}")
    print()
    print("one decoder layer:")
    print(lm.arch.layers[0])


def chat(lm):
    tok, model = lm.tok, lm.model
    history = []
    print(f"Chatting with {lm.id} on {lm.device}. /reset to clear, /quit to exit.\n")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/quit":
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)\n")
            continue

        history.append({"role": "user", "content": user})
        inputs = tok.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(lm.device)

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=512, do_sample=True,
                          temperature=0.7, top_p=0.9)
        Thread(target=model.generate, kwargs=gen_kwargs).start()

        print("model> ", end="", flush=True)
        reply = ""
        for chunk in streamer:
            print(chunk, end="", flush=True)
            reply += chunk
        print("\n")
        history.append({"role": "assistant", "content": reply})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id")
    parser.add_argument("--info", action="store_true", help="print architecture and exit")
    args = parser.parse_args()

    lm = LoadedModel(args.model)
    if args.info:
        print_info(lm)
    else:
        chat(lm)


if __name__ == "__main__":
    main()
