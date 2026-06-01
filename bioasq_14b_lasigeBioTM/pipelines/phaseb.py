
import os
import re
import json
import pickle
import traceback
import argparse

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "google/gemma-4-E4B-it" #or Intelligent-Internet/II-Medical-8B

TRAIN_DATA = "" 
TEST_DATA= ""
OUTPUT_DIR= ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--fresh", action="store_true",
                    help="Ignore any existing checkpoint and start from scratch")
args = parser.parse_args()

N_PROMPT= 5 #few-shot examples per question type

MAX_TOKENS = {
    "yesno": 10, 
    "factoid": 120,
    "list": 300,
    "summary": 350, 
}

_tag = f"submission_phaseB_batch4_fewshots{N_PROMPT}_gemma4"
PICKLE_FILE = os.path.join(OUTPUT_DIR, f"{_tag}.pkl")
CSV_FILE= os.path.join(OUTPUT_DIR, f"{_tag}.csv")
JSON_FILE= os.path.join(OUTPUT_DIR, f"{_tag}.json")


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"\nLoading {MODEL_ID} …")
tokenizer = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.generation_config.pad_token_id = tokenizer.tokenizer.eos_token_id
print("Model loaded.\n")

def process_jsonl_line(line: dict) -> list:
   
    context = "\n\n".join(s["text"] for s in line["snippets"])
    base = {"id": line["id"], "body": line["body"], "context": context}

    def summary_record(q_type_key: str, exact_a: str="") -> dict:

            ideal = line.get("ideal_answer")
            ideal_text = ideal[0] if ideal else "" 

            return {
            **base,
            "type": q_type_key,
            "answer": ideal_text,
            "exact_answer": exact_a}

    q_type = line["type"]

    if q_type == "yesno":
        exact_a=line["exact_answer"]
        return [
            {**base, "type": "yesno", "answer": exact_a},
            summary_record("yesno_summary", exact_a)
        ]
    elif q_type == "list":
        exact_str = ('{"entities": [' + ", ".join(f'"{e[0]}"' for e in line["exact_answer"]) + ']}')
        return [
            {**base, "type": "list", "answer": exact_str},
            summary_record("list_summary", exact_str)
        ]
    elif q_type == "factoid":
        exact_str = '{"entities": [' + ", ".join(f'"{e}"' for e in line["exact_answer"]) + ']}'
        return [
            {**base, "type": "factoid", "answer": exact_str},
            summary_record("factoid_summary", exact_str)
        ]
    elif q_type == "summary":
        return [summary_record("summary")]

    return []


def process_jsonl_line_new(line: dict) -> list:
    
    context = "\n\n".join(s["text"] for s in line["snippets"])
    base = {"id": line["id"], "body": line["body"], "context": context}

    def summary_record_new(q_type_key: str) -> dict:
        return {**base, "type": q_type_key, "answer": ""}

    q_type = line["type"]

    if q_type == "yesno":
        return [
            {**base, "type": "yesno", "answer": ""},
            summary_record_new("yesno_summary")
        ]
    elif q_type == "list":
        return [
            {**base, "type": "list", "answer": ""},
            summary_record_new("list_summary")
        ]
    elif q_type == "factoid":
        return [
            {**base, "type": "factoid", "answer": ""},
            summary_record_new("factoid_summary")
        ]
    elif q_type == "summary":
        return [summary_record_new("summary")]

    return []


def load_and_group_train(filepath: str) -> dict:
    buckets = {
        "yesno": [], "factoid": [], "list": [], "summary": [],
        "yesno_summary": [], "factoid_summary": [], "list_summary": []
    }
    with open(filepath) as f:
        for raw in f:
            for rec in process_jsonl_line(json.loads(raw)):
                if rec["type"] in buckets:
                    buckets[rec["type"]].append(rec)

    print("Few-shot pool sizes:")
    for k, v in buckets.items():
        print(f"  {k:20s}: {len(v)}")
    return buckets


print("Loading training data …")
grouped_train = load_and_group_train(TRAIN_DATA)


with open(TEST_DATA) as f:
    test_data = json.load(f)
questions = test_data["questions"]
print(f"\nLoaded {len(questions)} test questions.")


SYSTEM_PROMPT = (
    "You are an expert biomedical question-answering assistant. "
    "Answer questions using the provided passages and your medical knowledge. "
    "Always return the answer STRICTLY in the requested format."
)

def format_instruction(q_type: str) -> str:
    if q_type == "yesno":
        return ("You must answer ONLY with lowercase 'yes' or 'no', even if you are unsure.")
    if q_type == "list":
        return (
            "Your response MUST be a single JSON object and nothing else.\n"
            "Use EXACTLY this format: {\"entities\": [\"answer1\", \"answer2\"]}\n"
            " Rules:\n Up to 100 entries, max 100 characters each.\n"
            " Answers may be supplemented with your own knowledge if needed.\n"
            " Prefer the shortest, cleanest, and most direct entry cut from the Passage that accurately answers the question.\n"
            " Note that all phrases, even if not complete, are related to the question.\n"
            " Do not include synonyms or duplicate meanings.\n"
            " If unknown, return an empty array.\n"
            'Example: {"entities": ["entity1", "entity2"]}')
    
    if q_type == "factoid":
        return (
            "Your response MUST be a single JSON object and nothing else.\n"
            "Use EXACTLY this format: {\"entities\": [\"answer1\", \"answer2\"]}\n"
            " Rules:\n"
            " - Include ALL plausible answers, up to 5, ordered by decreasing confidence.\n"
            " - Do NOT stop at one answer if multiple valid answers exist.\n"
            " - Each entry must be a short, direct answer (a name, number, or expression).\n"
            " - Do not include synonyms or duplicate meanings.\n"
            " - Avoid acronyms unless the full name is unavailable.\n"
            " - When multiple numerical answers exist, merge into a single range.\n"
            " - If truly unknown, return an empty array: {\"entities\": []}\n"
            'Example: {"entities": ["Metformin", "Insulin", "Glipizide"]}'
        )
    return (
        "Please give a concise and correct answer to the question. STRICT LIMIT: your answer must be at most 200 words.\n"
        " Rules:\n Prefer answers that only combine direct phrases or sentences from the Passage.\n"
        " You may add minimal connecting words (such as 'and', 'which','that', 'because', 'thus') to make the answer grammatically correct.\n"
        " Do NOT start your answer with phrases like 'Based on the passage' or 'The answer is'. Go straight to the answer."
    )

def few_shot_messages(example: dict) -> list:
    """One (user, assistant) pair for a training example."""
    return [
        {
            "role": "user",
            "content": (
                f"Passage:\n{example['context']}\n\n"
                f"Question: {example['body']}\n\n"
                f"{format_instruction(example['type'])}"
            )
        },
        {"role": "assistant", "content": str(example["answer"])}
    ]

def few_shot_messages_ideal(example: dict) -> list:
    """
    Embeds the exact answer as a hint in the user turn so the model learns to write ideal answer.
    """
    hint = (
        f" (Hint: {example['exact_answer']})"
        if example.get("exact_answer")
        else ""
    )
    return [
        {
            "role": "user",
            "content": (
                f"Passage:\n{example['context']}\n\n"
                f"Question: {example['body']}{hint}\n\n"
                f"{format_instruction(example['type'])}"
            )
        },
        {"role": "assistant", "content": str(example["answer"])}
    ]

def build_prompt(question: dict, pool: list, n: int) -> list:
    """
    Assemble the full chat prompts:
      [system] + n × [user, assistant] + [user (the actual question)]
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in pool[:n]:
        messages += few_shot_messages(ex)
    messages.append({
        "role": "user",
        "content": (
            f"Passage:\n{question['context']}\n\n"
            f"Question: {question['body']}\n\n"
            f"{format_instruction(question['type'])}"
        ),
    })
    return messages

def build_ideal_prompt(question: dict, pool: list, n: int, exact_hint: str = "") -> list:
    """ 
    Few-shot examples use few_shot_messages_ideal (hint embedded).
    The real question also receives the exact_hint if one was generated.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in pool[:n]:
        messages += few_shot_messages_ideal(ex)
    hint_str = f" (Hint: {exact_hint})" if exact_hint else ""
    messages.append({
        "role": "user",
        "content": (
            f"Passage:\n{question['context']}\n\n"
            f"Question: {question['body']}{hint_str}\n\n"
            f"{format_instruction(question['type'])}"
        ),
    })
    return messages


def run_inference(messages: list, max_new_tokens: int) -> str:

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        #enable_thinking=True,   
    )
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.tokenizer.eos_token_id
        )

    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _extract_entity_list(text: str) -> list:
 
    match = re.search(r"\[([^\]]+)\]", text)
    if not match:
        return []
    return [
        item.strip().strip('"').strip("'")
        for item in match.group(1).split(",")
        if item.strip().strip('"').strip("'")]

def _strip_reasoning(text: str) -> str:
    """
    Kept as a fallback in case any residual tags appear with gemma4 (or II-Medical-8B with different tags) not really used.
    """
    channel_end = text.find("<channel|>")
    if channel_end != -1:
        text = text[channel_end + len("<channel|>"):]

    think_start = text.find("<think>")
    think_end = text.find("</think>")
    if think_start != -1 and think_end != -1:
        text = text[:think_start] + text[think_end + len("</think>"):]

    return text.strip()


def clean_answer(raw: str, q_type: str):
    """
    BioASQ submission format:
      yesno: "yes" or "no"
      factoid: ["e1", "e2", ...]
      list: [["e1"], ["e2"], ...] 
      summary: str
    """
    text = _strip_reasoning(raw).strip()

    if q_type == "yesno":
        lower = text.lower()
        return "yes" if "yes" in lower else ("no" if "no" in lower else "yes") #on training "yes" is 70% of the time (not really used, but a fallback)

    if q_type == "factoid":
        m = re.search(r'"entities"\s*:\s*(\[[^\]]*\])', text, re.IGNORECASE)
        entities = _extract_entity_list(m.group(1) if m else text)
        return [[e] for e in entities if len(e) > 1][:5]

    if q_type == "list":
        m = re.search(r'"entities"\s*:\s*(\[[^\]]*\])', text, re.IGNORECASE)
        entities = _extract_entity_list(m.group(1) if m else text)
        return [[e] for e in entities if len(e) > 2][:100]

    for prefix in ("answer:", "concise answer:", "here is", "here's", "based on the passage", "based on the provided"):
        idx = text.lower().find(prefix)
        if idx != -1:
            text = text[idx + len(prefix):].lstrip(" :\n")
    words=text.split()
    if len(words)>200:
        truncated=" ".join(words[:200])
        last_period=max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        text=truncated[:last_period+1] if last_period > 100 else truncated
    return text

def get_answer(q: dict) -> dict:
    """
    Dual-pass approach. First pass generates the exact answer and then the second pass embeddes that exact answer and asks the model for the ideal answer.
    """
    processed = process_jsonl_line_new(q)
    q_type = q["type"]
    
    if q_type != "summary": 
        exact_prompt = build_prompt(
            processed[0], grouped_train[q_type], N_PROMPT
        )
        raw_exact = run_inference(exact_prompt, MAX_TOKENS[q_type])
        exact_answer = clean_answer(raw_exact, q_type)
 
        if isinstance(exact_answer, list):
            hint_str = ", ".join(
                e[0] if isinstance(e, list) else str(e)
                for e in exact_answer
            )
        else:
            hint_str = str(exact_answer)
 
        ideal_prompt = build_ideal_prompt(
            processed[1],
            grouped_train[q_type + "_summary"],
            N_PROMPT,
            exact_hint=hint_str
        )
        raw_ideal = run_inference(ideal_prompt, MAX_TOKENS["summary"])
        ideal_answer = clean_answer(raw_ideal, "summary")
 
    else:
        exact_answer = ""
        #no second pass for summary questions (they only require ideal answer)
        ideal_prompt = build_ideal_prompt(
            processed[0], grouped_train["summary"], N_PROMPT
        )
        raw_ideal = run_inference(ideal_prompt, MAX_TOKENS["summary"])
        ideal_answer = clean_answer(raw_ideal, "summary")
        if not ideal_answer or not ideal_answer.strip():
            ideal_answer = "No answer generated."
 
    # Fallback for the II-Medical-8B mainly
    # This rescues cases where the model wrote prose instead of JSON.
    if q_type in ("factoid", "list") and not exact_answer and ideal_answer and \
            ideal_answer != "No answer generated.":
        fallback = clean_answer(ideal_answer, q_type)
        if fallback:
            exact_answer = fallback

    return {
        "id": q["id"],
        "type": q_type,
        "body": q["body"],
        "documents": q["documents"],
        "snippets": q["snippets"],
        "ideal_answer": ideal_answer,
        "exact_answer": exact_answer
    }

#Checkpointing
def save_checkpoint(df: pd.DataFrame):
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(df, f)


def load_checkpoint() -> pd.DataFrame | None:
    if not os.path.exists(PICKLE_FILE):
        return None
    try:
        with open(PICKLE_FILE, "rb") as f:
            return pickle.load(f)
    except EOFError:
        return None


#Main loop
if args.fresh and os.path.exists(PICKLE_FILE):
    os.remove(PICKLE_FILE)
    print("--fresh flag set: deleted existing checkpoint, starting from scratch.")

saved_df = load_checkpoint()

if saved_df is not None and not saved_df.empty:
    processed_ids = set(saved_df["id"])
    questions_df  = saved_df
    print(f"Resuming: {len(processed_ids)} questions already processed.")
else:
    processed_ids = set()
    questions_df  = pd.DataFrame(
        columns=["id", "body", "type", "documents",
                 "snippets", "ideal_answer", "exact_answer"]
    )
    print("Starting from scratch.")

questions_to_process = [q for q in questions if q["id"] not in processed_ids]
print(f"Questions to process: {len(questions_to_process)}\n")

for i, q in enumerate(questions_to_process, 1):
        try:
            result = get_answer(q)
            questions_df = pd.concat(
                [questions_df, pd.DataFrame([result])],
                ignore_index=True)
            save_checkpoint(questions_df)
            print(f"[{i:3d}/{len(questions_to_process)}] {q['id']}  ({q['type']})  Check!!")
        except Exception as exc:
            print(f"[{i:3d}/{len(questions_to_process)}] ERROR — {q['id']}: {exc}")
            traceback.print_exc()

questions_df.to_csv(CSV_FILE, index=False)
print(f"\nCSV saved → {CSV_FILE}")

def build_submission_json(csv_path: str, json_path: str):
    
    df = pd.read_csv(csv_path)
    output = {"questions": []}

    for _, row in df.iterrows():
        q_type = row["type"]
        
        ideal = row["ideal_answer"]
        if isinstance(ideal, list):
            ideal = ideal[0] if ideal else ""
        ideal = str(ideal)
        if ideal.strip().lower() == "nan" or ideal.strip() == "":
            ideal = "No answer generated."

        entry  = {
            "id": row["id"],
            "body": row["body"],
            "type": q_type,
            "documents": eval(row["documents"])[:10],
            "snippets": eval(row["snippets"])[:10],
            "ideal_answer": ideal}

        if q_type == "yesno":
            ans = str(row["exact_answer"]).strip().lower()
            entry["exact_answer"] = ans

        elif q_type == "factoid":
            raw = eval(row["exact_answer"])

            entry["exact_answer"] = [item if isinstance(item, list) else [item] for item in raw][:5]

        elif q_type == "list":
            raw = eval(row["exact_answer"])
            entry["exact_answer"] = [item if isinstance(item, list) else [item] for item in raw][:100]

        output["questions"].append(entry)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"Submission JSON → {json_path}  ({len(output['questions'])} questions)")


build_submission_json(CSV_FILE, JSON_FILE)
print(f"\nReady to submit: {JSON_FILE}")