import torch
import argparse
import os
import json
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
from openai import OpenAI  # or swap for another LLM client

from dataset_cfg import (
    ground_truth, dataset_root, image_paths, hf_model_name,
    augmented_prompts_obj, augmented_prompts_task, augmented_prompts_task_obj,
    chain_of_thought
)
from plotter import plot_results

# ---------------------------------------------------------------------------
# LLM client setup
# Set your API key via environment variable: export OPENAI_API_KEY=sk-...
# Or swap this out for HuggingFace / Anthropic / Ollama client as needed
# ---------------------------------------------------------------------------
llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
LLM_MODEL = "gpt-3.5-turbo"  # swap for "gpt-4", "gpt-4o", etc.


# ---------------------------------------------------------------------------
# CLIP: return full scores for all candidates instead of just the top pick
# ---------------------------------------------------------------------------
def run_clip_scores(model, processor, text, images, names, device):
    """
    Returns a dict of {object_name: similarity_score} for all candidates.
    This replaces run_clip_eval from the original code.
    """
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_image shape: (num_images, 1) — similarity of each image to the text
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=0).squeeze()  # normalize across candidates

    scores = {name: round(prob.item(), 4) for name, prob in zip(names, probs)}
    return scores


# ---------------------------------------------------------------------------
# LLM CoT: takes CLIP scores + affordance context, reasons, picks best object
# ---------------------------------------------------------------------------
def run_llm_cot(tool_name, affordance_description, candidate_scores):
    """
    Sends CLIP scores + affordance context to an LLM.
    The LLM performs chain-of-thought reasoning and returns the best candidate.

    Args:
        tool_name: e.g. "hammer"
        affordance_description: the affordance prompt string from dataset_cfg.py
        candidate_scores: dict of {object_name: clip_score}

    Returns:
        predicted_object: string name of the LLM's final selection
        reasoning: the LLM's full CoT reasoning text
    """
    # Format candidate scores for the prompt
    candidates_str = "\n".join(
        f"- {name}: {score:.4f}" for name, score in sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
    )

    prompt = f"""You are helping identify the best substitute object for a missing tool.

Missing tool: {tool_name}
Tool description: {affordance_description}

A vision model (CLIP) has scored the following candidate objects based on how 
visually similar they are to the tool description (higher = more similar):

{candidates_str}

Using chain-of-thought reasoning, please:
1. Identify the core physical requirements of a {tool_name}
2. Evaluate each candidate object against these requirements, considering both 
   the CLIP scores and your knowledge of the objects
3. Select the single best substitute object

Format your response as:
REASONING: <your step-by-step analysis>
SELECTION: <object name, exactly as listed above>"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,  # deterministic for reproducibility
        max_tokens=500
    )

    response_text = response.choices[0].message.content.strip()

    # Parse the selection from the response
    predicted_object = parse_llm_selection(response_text, candidate_scores)
    return predicted_object, response_text


def parse_llm_selection(response_text, candidate_scores):
    """
    Extracts the selected object name from the LLM response.
    Falls back to highest CLIP score if parsing fails.
    """
    lines = response_text.strip().split("\n")
    for line in lines:
        if line.startswith("SELECTION:"):
            selection = line.replace("SELECTION:", "").strip().lower()
            # Match against known candidate names (fuzzy match)
            for name in candidate_scores:
                if name.lower() in selection or selection in name.lower():
                    return name

    # Fallback: return highest CLIP score candidate
    print("[Warning] Could not parse LLM selection, falling back to CLIP top pick")
    return max(candidate_scores, key=candidate_scores.get)


# ---------------------------------------------------------------------------
# Model loader (CLIP only — ViLT excluded due to token limit constraints)
# ---------------------------------------------------------------------------
def get_clip_model(model_name, device):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, processor


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def main(model_name, args):
    random.seed(args.seed)

    def create_random_three_objects(image_paths, ground_truth_obj, exclude=""):
        objects = [k for k in image_paths.keys() if k != ground_truth_obj and k != exclude]
        random.shuffle(objects)
        return [ground_truth_obj] + objects[:3]

    def get_accuracy(text, predicted_object, ground_truth_map):
        for obj in ground_truth_map.keys():
            if obj in text:
                return 1 if ground_truth_map[obj] == predicted_object else 0
        return 0

    mode = args.task_type
    image_full_paths = {k: dataset_root + "/" + v for k, v in image_paths.items()}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, processor = get_clip_model(model_name, device)

    accuracy = 0
    accuracy_by_class = {}
    reasoning_log = []  # store CoT reasoning for analysis
    N_range = 10
    N_tasks = 5

    # Map from prompt text to its affordance description (for LLM context)
    # TODO: update this mapping if you add new prompt types
    prompt_to_tool = {
        augmented_prompts_obj[0]: ("scoop", "concave and hollow, used to transfer materials"),
        augmented_prompts_obj[1]: ("hammer", "heavy, handle attached to a cylinder at the end"),
        augmented_prompts_obj[2]: ("spatula", "handle attached to a flat surface at the end"),
        augmented_prompts_obj[3]: ("toothpick", "pointed tip, used to pick food between teeth"),
        augmented_prompts_obj[4]: ("pliers", "two-pronged, used to grip objects"),
    }

    for _ in tqdm(range(N_range)):
        # Build test sets (same as original eval_task.py)
        dataset_mapping = {
            "creative": {
                "can this object be used as a scoop?": create_random_three_objects(image_paths, "bowl", exclude="spoon"),
                "can this object be used as a hammer?": create_random_three_objects(image_paths, "saucepan", exclude="hammer"),
                "can this object be used as a spatula?": create_random_three_objects(image_paths, "knife", exclude="spatula"),
                "can this object be used as a toothpick?": create_random_three_objects(image_paths, "safety pin", exclude="toothpick"),
                "can this object be used as pliers?": create_random_three_objects(image_paths, "scissors", exclude="pliers"),
            }
        }

        if mode == "creative-obj-cot":
            dataset_mapping["creative-obj-cot"] = {
                k: v for k, v in zip(augmented_prompts_obj, dataset_mapping["creative"].values())
            }
            text_list = [t for t in dataset_mapping["creative-obj-cot"]]
            current_gt = ground_truth["creative-obj"]
        else:
            # TODO: add other modes (creative-task-cot, creative-task-obj-cot) here
            raise ValueError(f"Unsupported mode: {mode}")

        for text in text_list:
            # Load candidate images
            images, names = [], []
            for name, path in image_full_paths.items():
                if name in dataset_mapping[mode][text]:
                    images.append(Image.open(path))
                    names.append(name)

            # Step 1: Get CLIP similarity scores for all candidates
            candidate_scores = run_clip_scores(model, processor, text, images, names, device)

            if args.verbose:
                print(f"\n[CLIP Scores] {text}")
                for n, s in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {n}: {s:.4f}")

            # Step 2: Pass scores + context to LLM for CoT reasoning
            tool_name, affordance_desc = prompt_to_tool.get(text, ("unknown tool", text))
            predicted_object, reasoning = run_llm_cot(tool_name, affordance_desc, candidate_scores)

            if args.verbose:
                print(f"[LLM Selection] {predicted_object}")
                print(f"[LLM Reasoning]\n{reasoning}\n")

            # Log for later analysis
            reasoning_log.append({
                "text": text,
                "clip_scores": candidate_scores,
                "llm_selection": predicted_object,
                "llm_reasoning": reasoning,
                "correct": get_accuracy(text, predicted_object, current_gt)
            })

            acc = get_accuracy(text, predicted_object, current_gt)
            accuracy += acc
            accuracy_by_class[text] = accuracy_by_class.get(text, 0) + acc

    # Save reasoning log for qualitative analysis
    if args.save_reasoning:
        with open(f"reasoning_log_{mode}.json", "w") as f:
            json.dump(reasoning_log, f, indent=2)
        print(f"Reasoning log saved to reasoning_log_{mode}.json")

    accuracy_by_class = {k: v / N_range for k, v in accuracy_by_class.items()}
    overall = np.mean(list(accuracy_by_class.values()))
    accuracy_by_class["overall"] = overall
    return accuracy_by_class


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLIP + LLM Chain-of-Thought pipeline for creative object substitution"
    )
    parser.add_argument(
        "--task-type", type=str, required=True,
        help="Prompt type: creative-obj-cot | creative-task-cot | creative-task-obj-cot"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--save-reasoning", action="store_true",
        help="Save LLM CoT reasoning to a JSON file for qualitative analysis"
    )
    args = parser.parse_args()

    plotting_data = {}
    for name, hf_name in hf_model_name.items():
        print(f"\n=== Model: {name} ===")
        acc_by_class = main(hf_name, args)
        plotting_data[name] = acc_by_class
        print(f"Overall accuracy: {acc_by_class['overall']:.2f}")

    print("\nSaving visualization...")
    plot_results(args.task_type, plotting_data)