"""
Example for calculating metrics, with validation that HF checkpoints produce the same logits as released results
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from hf_olmo import OLMoForCausalLM
from datasets import load_dataset

# Configuration
MODEL_NAME = "allenai/DataDecide-dolma1_7-150M"
MODEL_REVISION = "step38750-seed-small-aux-3"
NUM_SHOT = 5
TOLERANCE = 1e-3

# Reference results from hf: allenai/DataDecide-eval-instances/models/dolma1.7/150M/seed-15/step-38750
REFERENCE_RESULTS = [
    {"doc_id": 0, "native_id": "Mercury_417466", "metrics": {"predicted_index_raw": 2, "predicted_index_per_token": 0, "predicted_index_per_char": 0, "predicted_index_uncond": 3, "correct_choice": 0, "acc_raw": 0, "acc_per_token": 1, "acc_per_char": 1, "acc_uncond": 0}, "model_output": [{"sum_logits": -33.75307846069336, "num_tokens": 12, "num_tokens_all": 186, "is_greedy": False, "sum_logits_uncond": -44.89948272705078, "logits_per_token": -2.8127565383911133, "logits_per_char": -0.5625513076782227, "logits_per_byte": 0.8115899818335565, "num_chars": 60}, {"sum_logits": -36.5004768371582, "num_tokens": 11, "num_tokens_all": 185, "is_greedy": False, "sum_logits_uncond": -42.012271881103516, "logits_per_token": -3.3182251670143823, "logits_per_char": -0.6636450334028764, "logits_per_byte": 0.9574373986016081, "num_chars": 55}, {"sum_logits": -29.177223205566406, "num_tokens": 9, "num_tokens_all": 183, "is_greedy": False, "sum_logits_uncond": -36.19854736328125, "logits_per_token": -3.2419136895073786, "logits_per_char": -0.5721024157954198, "logits_per_byte": 0.8253693181492409, "num_chars": 51}, {"sum_logits": -38.646724700927734, "num_tokens": 8, "num_tokens_all": 182, "is_greedy": False, "sum_logits_uncond": -50.06067657470703, "logits_per_token": -4.830840587615967, "logits_per_char": -0.8783346522938121, "logits_per_byte": 1.2671690471061252, "num_chars": 44}], "label": 0, "task_hash": "c90b1e74fd08b92cb197198ea6975132", "model_hash": "32890bd27f744d39a1855746ba84775f"},
    {"doc_id": 1, "native_id": "Mercury_7081673", "metrics": {"predicted_index_raw": 1, "predicted_index_per_token": 0, "predicted_index_per_char": 1, "predicted_index_uncond": 1, "correct_choice": 1, "acc_raw": 1, "acc_per_token": 0, "acc_per_char": 1, "acc_uncond": 1}, "model_output": [{"sum_logits": -12.96231460571289, "num_tokens": 4, "num_tokens_all": 180, "is_greedy": False, "sum_logits_uncond": -22.333494186401367, "logits_per_token": -3.2405786514282227, "logits_per_char": -0.864154307047526, "logits_per_byte": 1.2467111333412002, "num_chars": 15}, {"sum_logits": -10.86353874206543, "num_tokens": 2, "num_tokens_all": 178, "is_greedy": False, "sum_logits_uncond": -21.388111114501953, "logits_per_token": -5.431769371032715, "logits_per_char": -0.7242359161376953, "logits_per_byte": 1.0448515646462788, "num_chars": 15}, {"sum_logits": -11.422426223754883, "num_tokens": 2, "num_tokens_all": 178, "is_greedy": False, "sum_logits_uncond": -17.692312240600586, "logits_per_token": -5.711213111877441, "logits_per_char": -0.815887587411063, "logits_per_byte": 1.1770769762816469, "num_chars": 14}, {"sum_logits": -19.420806884765625, "num_tokens": 3, "num_tokens_all": 179, "is_greedy": False, "sum_logits_uncond": -21.712383270263672, "logits_per_token": -6.473602294921875, "logits_per_char": -1.7655278986150569, "logits_per_byte": 2.5471183438848852, "num_chars": 11}], "label": 1, "task_hash": "c90b1e74fd08b92cb197198ea6975132", "model_hash": "32890bd27f744d39a1855746ba84775f"},
    {"doc_id": 2, "native_id": "Mercury_7239733", "metrics": {"predicted_index_raw": 2, "predicted_index_per_token": 2, "predicted_index_per_char": 2, "predicted_index_uncond": 3, "correct_choice": 3, "acc_raw": 0, "acc_per_token": 0, "acc_per_char": 0, "acc_uncond": 1}, "model_output": [{"sum_logits": -14.236495971679688, "num_tokens": 2, "num_tokens_all": 186, "is_greedy": False, "sum_logits_uncond": -15.521793365478516, "logits_per_token": -7.118247985839844, "logits_per_char": -1.1863746643066406, "logits_per_byte": 1.7115768448327289, "num_chars": 12}, {"sum_logits": -14.386190414428711, "num_tokens": 2, "num_tokens_all": 186, "is_greedy": False, "sum_logits_uncond": -20.770462036132812, "logits_per_token": -7.1930952072143555, "logits_per_char": -1.307835492220792, "logits_per_byte": 1.8868077789268687, "num_chars": 11}, {"sum_logits": -12.404546737670898, "num_tokens": 2, "num_tokens_all": 186, "is_greedy": False, "sum_logits_uncond": -17.448867797851562, "logits_per_token": -6.202273368835449, "logits_per_char": -0.9541959028977615, "logits_per_byte": 1.3766136971481564, "num_chars": 13}, {"sum_logits": -14.411821365356445, "num_tokens": 2, "num_tokens_all": 186, "is_greedy": False, "sum_logits_uncond": -21.861894607543945, "logits_per_token": -7.205910682678223, "logits_per_char": -1.2009851137797039, "logits_per_byte": 1.732655267832691, "num_chars": 12}], "label": 3, "task_hash": "c90b1e74fd08b92cb197198ea6975132", "model_hash": "32890bd27f744d39a1855746ba84775f"},
    {"doc_id": 3, "native_id": "NYSEDREGENTS_2015_4_8", "metrics": {"predicted_index_raw": 3, "predicted_index_per_token": 3, "predicted_index_per_char": 3, "predicted_index_uncond": 3, "correct_choice": 3, "acc_raw": 1, "acc_per_token": 1, "acc_per_char": 1, "acc_uncond": 1}, "model_output": [{"sum_logits": -10.329045295715332, "num_tokens": 1, "num_tokens_all": 172, "is_greedy": False, "sum_logits_uncond": -12.410356521606445, "logits_per_token": -10.329045295715332, "logits_per_char": -2.0658090591430662, "logits_per_byte": 2.980332485051338, "num_chars": 5}, {"sum_logits": -8.12256908416748, "num_tokens": 1, "num_tokens_all": 172, "is_greedy": False, "sum_logits_uncond": -11.736161231994629, "logits_per_token": -8.12256908416748, "logits_per_char": -1.624513816833496, "logits_per_byte": 2.3436780274029703, "num_chars": 5}, {"sum_logits": -6.024718284606934, "num_tokens": 1, "num_tokens_all": 172, "is_greedy": False, "sum_logits_uncond": -9.307938575744629, "logits_per_token": -6.024718284606934, "logits_per_char": -1.2049436569213867, "logits_per_byte": 1.738366238392346, "num_chars": 5}, {"sum_logits": -5.648414611816406, "num_tokens": 1, "num_tokens_all": 172, "is_greedy": False, "sum_logits_uncond": -12.483970642089844, "logits_per_token": -5.648414611816406, "logits_per_char": -1.1296829223632812, "logits_per_byte": 1.6297879498716286, "num_chars": 5}], "label": 3, "task_hash": "c90b1e74fd08b92cb197198ea6975132", "model_hash": "32890bd27f744d39a1855746ba84775f"},
    {"doc_id": 4, "native_id": "Mercury_7037258", "metrics": {"predicted_index_raw": 2, "predicted_index_per_token": 2, "predicted_index_per_char": 2, "predicted_index_uncond": 1, "correct_choice": 1, "acc_raw": 0, "acc_per_token": 0, "acc_per_char": 0, "acc_uncond": 1}, "model_output": [{"sum_logits": -41.315250396728516, "num_tokens": 7, "num_tokens_all": 175, "is_greedy": False, "sum_logits_uncond": -52.73341369628906, "logits_per_token": -5.902178628104074, "logits_per_char": -0.8101029489554611, "logits_per_byte": 1.1687315070684083, "num_chars": 51}, {"sum_logits": -39.29490661621094, "num_tokens": 9, "num_tokens_all": 177, "is_greedy": False, "sum_logits_uncond": -54.706912994384766, "logits_per_token": -4.366100735134548, "logits_per_char": -0.6774983899346714, "logits_per_byte": 0.9774235673697098, "num_chars": 58}, {"sum_logits": -30.93582534790039, "num_tokens": 8, "num_tokens_all": 176, "is_greedy": False, "sum_logits_uncond": -42.650123596191406, "logits_per_token": -3.866978168487549, "logits_per_char": -0.5427337780333402, "logits_per_byte": 0.7829993300921939, "num_chars": 57}, {"sum_logits": -44.936012268066406, "num_tokens": 10, "num_tokens_all": 178, "is_greedy": False, "sum_logits_uncond": -50.64436340332031, "logits_per_token": -4.493601226806641, "logits_per_char": -0.7021251916885376, "logits_per_byte": 1.0129525321329937, "num_chars": 64}], "label": 1, "task_hash": "c90b1e74fd08b92cb197198ea6975132", "model_hash": "32890bd27f744d39a1855746ba84775f"},
]

# Few-shot examples from OLMo-Eval public
FEWSHOT_EXAMPLES = {
    "ARC-Easy": [
        {
            "id": "MCAS_2007_8_5189",
            "question": "Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?",
            "choices": {
                "text": ["carbon dioxide", "food", "protection", "water"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_SC_401169",
            "question": "When a switch is used in an electrical circuit, the switch can",
            "choices": {
                "text": [
                    "cause the charge to build.",
                    "increase and decrease the voltage.",
                    "cause the current to change direction.",
                    "stop and start the flow of current.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MCAS_2004_8_27",
            "question": "Which of the following is an example of an assistive device?",
            "choices": {
                "text": ["contact lens", "motorcycle", "raincoat", "coffee pot"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "NYSEDREGENTS_2006_8_10",
            "question": "Rocks are classified as igneous, metamorphic, or sedimentary according to",
            "choices": {
                "text": ["their color", "their shape", "how they formed", "the minerals they contain"],
                "label": ["1", "2", "3", "4"],
            },
            "answerKey": "3",
        },
        {
            "id": "Mercury_7013388",
            "question": "A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium carbonate is most likely used as this type of medicine because calcium carbonate",
            "choices": {
                "text": [
                    "has a pleasant flavor.",
                    "is inexpensive to produce.",
                    "neutralizes digestive acid.",
                    "occurs naturally in the body.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "Mercury_7179953",
            "question": "Which two body systems are directly involved in movement?",
            "choices": {
                "text": [
                    "muscular and skeletal",
                    "digestive and muscular",
                    "skeletal and respiratory",
                    "respiratory and digestive",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "Mercury_7205118",
            "question": "Which change in the state of water particles causes the particles to become arranged in a fixed position?",
            "choices": {
                "text": ["boiling", "melting", "freezing", "evaporating"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "MCAS_2016_8_13",
            "question": "Earth's core is primarily composed of which of the following materials?",
            "choices": {"text": ["basalt", "iron", "magma", "quartz"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
    ]
}


def build_fewshot_prefix(fewshot_docs):
    """Build few-shot prompt prefix from example documents."""
    blocks = []
    for doc in fewshot_docs:
        formatted = format_arc_ranked(doc)
        correct_label = formatted["correct_choice"]
        label_to_text = dict(zip(formatted["choices"], formatted["choice_texts"]))
        blocks.append(f"Question: {doc['question']}\nAnswer:{label_to_text[correct_label]}")
    return "\n\n".join(blocks)


def evaluate_sample(example, fewshot_prefix):
    """Evaluate a single sample and return prediction metrics."""
    example["prompt"] = fewshot_prefix + "\n\n" + example["prompt"]
    
    model_output = [score_choice(example["prompt"], choice) for choice in example["choice_texts"]]
    
    gold = example["gold_index"]
    sum_logits = [x["sum_logits"] for x in model_output]
    logits_per_token = [x["logits_per_token"] for x in model_output]
    logits_per_char = [x["logits_per_char"] for x in model_output]
    
    pred_raw = int(np.argmax(sum_logits))
    pred_token = int(np.argmax(logits_per_token))
    pred_char = int(np.argmax(logits_per_char))
    
    return {
        "native_id": example["native_id"],
        "metrics": {
            "correct_choice": gold,
            "acc_raw": int(pred_raw == gold),
            "acc_per_token": int(pred_token == gold),
            "acc_per_char": int(pred_char == gold),
        },
        "model_output": model_output,
    }


def format_arc_ranked(doc):
    """Format ARC document for evaluation."""
    num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    doc["answerKey"] = num_to_letter.get(doc["answerKey"], doc["answerKey"])
    num_choices = len(doc["choices"]["text"])
    choice_labels = ["A", "B", "C", "D", "E"][:num_choices]
    question = doc["question"].strip()
    prompt = f"Question: {question}\nAnswer:"
    return {
        "native_id": doc["id"],
        "prompt": prompt,
        "choices": choice_labels,
        "choice_texts": [f" {t}" for t in doc["choices"]["text"]],
        "correct_choice": doc["answerKey"],
        "gold_index": choice_labels.index(doc["answerKey"])
    }


def score_choice(prompt, choice):
    """Calculate log probability scores for a choice continuation."""
    full_text = prompt + choice
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    full_inputs.pop("token_type_ids", None)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    full_ids = full_inputs["input_ids"][0]
    prompt_len = prompt_inputs["input_ids"].size(1)

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Scores for the tokens after the prompt
    target_ids = full_ids[prompt_len:]
    log_probs_choice = log_probs[0, prompt_len - 1 : -1]  # Shifted logprobs for next token
    token_logprobs = log_probs_choice.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return {
        "sum_logits": token_logprobs.sum().item(),
        "logits_per_token": token_logprobs.mean().item(),
        "logits_per_char": token_logprobs.sum().item() / len(choice),
        "logits_per_byte": token_logprobs.sum().item() / len(choice.encode("utf-8")),
        "num_tokens": len(token_logprobs),
        "num_chars": len(choice),
        "num_bytes": len(choice.encode("utf-8")),
    }


def compare_results(pred, ref):
    print(f"\nDoc ID: {pred['native_id']}")

    # Compare accuracies
    for key, pred_val in pred["metrics"].items():
        ref_val = ref["metrics"].get(key)
        print(f"{key}: predicted={pred_val}, reference={ref_val}")
        if pred_val != ref_val:
            print(f"[!] Mismatch: {pred_val} != {ref_val}")

    # Compare each answer logit
    for i, (p, r) in enumerate(zip(pred["model_output"], ref["model_output"])):
        for metric_name in ["sum_logits", "num_tokens", "num_chars"]:
            pred_val = p[metric_name]
            ref_val = r[metric_name]
            print(f"- {metric_name}: pred={pred_val:.2f}, ref={ref_val:.2f}")
            if not np.isclose(pred_val, ref_val, atol=TOLERANCE):
                print(f"[!] Choice {i} '{metric_name}' mismatch: {pred_val:.4f} != {ref_val:.4f}")

    print("-" * 40)


if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = OLMoForCausalLM.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).eval().to(device)
    
    # Load data
    dataset = load_dataset("ai2_arc", "ARC-Easy", split="test")
    fewshot_prefix = build_fewshot_prefix(FEWSHOT_EXAMPLES["ARC-Easy"][:NUM_SHOT])
    reference_lookup = {ref["native_id"]: ref for ref in REFERENCE_RESULTS}
    
    # Run validation
    for sample in dataset:
        if sample["id"] not in reference_lookup:
            continue
        
        formatted_example = format_arc_ranked(sample)
        pred = evaluate_sample(formatted_example, fewshot_prefix)
        compare_results(pred, reference_lookup[sample["id"]])