import torch
import argparse
import os
import json
from transformers import CLIPProcessor, CLIPModel
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import random
from dataset_cfg import ground_truth, dataset_root, image_paths, hf_model_name
from dataset_cfg import augmented_prompts_obj, augmented_prompts_task, augmented_prompts_task_obj, chain_of_thought
from plotter import plot_results
from tqdm import tqdm
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")


# LLM client setup for CoT
llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
LLM_MODEL = "gpt-3.5-turbo"  


def get_model(model_name):
    if "vilt" in model_name:
        processor = ViltProcessor.from_pretrained(model_name)
        model = ViltForQuestionAnswering.from_pretrained(model_name)
    else:
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)

    return model, processor


def run_llm_cot(tool_name, affordance_description, candidate_scores):
    """
    Sends vision scores + affordance context to an LLM.
    """
    candidates_str = "\n".join(
        f"- {name}: {score:.4f}" for name, score in sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
    )

    prompt = f"""You are helping identify the best substitute object for a missing tool.

Missing tool: {tool_name}
Tool description: {affordance_description}

A vision model has scored the following candidate objects based on how 
visually similar they are to the tool description (higher = more similar):

{candidates_str}

Using chain-of-thought reasoning, please:
1. Identify the core physical requirements of a {tool_name}
2. Evaluate each candidate object against these requirements, considering both 
   the vision model scores and your knowledge of the objects
3. Select the single best substitute object

Format your response as:
REASONING: <your step-by-step analysis>
SELECTION: <object name, exactly as listed above>"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, # deterministic for reproducibility
        max_tokens=500
    )

    response_text = response.choices[0].message.content.strip()
    # Parse the selection from the response
    predicted_object = parse_llm_selection(response_text, candidate_scores)
    return predicted_object, response_text


def parse_llm_selection(response_text, candidate_scores):
    """
    Extracts the selected object name from the LLM response.
    Falls back to highest vision score if parsing fails.
    """
    lines = response_text.strip().split("\n")
    for line in lines:
        if line.startswith("SELECTION:"):
            selection = line.replace("SELECTION:", "").strip().lower()
            for name in candidate_scores:
                if name.lower() in selection or selection in name.lower():
                    return name
    return max(candidate_scores, key=candidate_scores.get)


def run_vilt_eval(model, processor, text, images, names, device, return_scores=False):
    results = {}
    for i, img in enumerate(images):
        inputs = processor(img, text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits  # this is the image-text similarity score
        idx = logits.argmax(-1).item()
        predicted_answer = model.config.id2label[idx]
        if "yes" in predicted_answer.lower():
            results[names[i]] = logits.max(-1).values.item()

    if return_scores:
        # Fill missing names with 0.0 if ViLT didn't predict "yes"
        for name in names:
            if name not in results:
                results[name] = 0.0
        return results

    # Pick the key from results that has highest value
    if not results:
        predicted_object = "None"
    else:
        predicted_object = max(results, key=results.get)
    return predicted_object


def run_clip_eval(model, processor, text, images, names, device, return_scores=False):
    inputs = processor(text=text, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=0)
    
    if return_scores:
        scores_probs = probs.squeeze()
        if scores_probs.dim() == 0:
            scores_probs = [scores_probs.item()]
        else:
            scores_probs = scores_probs.tolist()
        return {name: round(prob, 4) for name, prob in zip(names, scores_probs)}

    idx = probs.argmax(dim=0)
    return names[idx]


def main(model_name, args):
    # Seed for reproducibility
    random.seed(args.seed)
    def create_random_three_objects(image_paths, ground_truth, exclude=""):
        objects = [k for k in image_paths.keys() if k != ground_truth and k != exclude]
        random.shuffle(objects)
        return [ground_truth] + objects[:3]

    def get_accuracy(text, predicted_object, ground_truth):
        for obj in ground_truth.keys():
            if obj in text:
                return 1 if ground_truth[obj] == predicted_object else 0
        return 0

    # CoT Prompt descriptors mapped directly to the tools in ground_truth keys
    tool_descriptions = {
        "scoop": "concave and hollow, used to transfer materials",
        "hammer": "heavy, handle attached to a cylinder at the end",
        "spatula": "handle attached to a flat surface at the end",
        "toothpick": "pointed tip, used to pick food between teeth",
        "pliers": "two-pronged, used to grip objects"
    }

    mode = args.task_type
    image_full_paths = {k: dataset_root + "/" + v for k, v in image_paths.items()}
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, processor = get_model(model_name)
    model.eval()
    model.to(device)

    accuracy = 0
    accuracy_by_class = {}
    reasoning_log = [] # store CoT reasoning for analysis
    N_range = 10  # Number of samples per task
    N_tasks = 5  # Number of tasks
    N_samples = N_range * N_tasks  # Total number of samples

    for _ in tqdm(range(N_range)):
        dataset_mapping = {
            "nominal": {
                "can this object be used as a scoop?": create_random_three_objects(image_paths, "spoon"),
                "can this object be used as a hammer?": create_random_three_objects(image_paths, "hammer"),
                "can this object be used as a spatula?": create_random_three_objects(image_paths, "spatula"),
                "can this object be used as a toothpick?": create_random_three_objects(image_paths, "toothpick"),
                "can this object be used as pliers?": create_random_three_objects(image_paths, "pliers")
            },
            "creative": {
                "can this object be used as a scoop?": create_random_three_objects(image_paths, "bowl", exclude="spoon"),
                "can this object be used as a hammer?": create_random_three_objects(image_paths, "saucepan", exclude="hammer"),
                "can this object be used as a spatula?": create_random_three_objects(image_paths, "knife", exclude="spatula"),
                "can this object be used as a toothpick?": create_random_three_objects(image_paths, "safety pin", exclude="toothpick"),
                "can this object be used as pliers?": create_random_three_objects(image_paths, "scissors", exclude="pliers")
            }
        }
        
        dataset_mapping["nominal-chain"] = {
            (k + " " + chain_of_thought): v for k, v in dataset_mapping["nominal"].items()
        }
        dataset_mapping["creative-chain"] = {
            (k + " " + chain_of_thought): v for k, v in dataset_mapping["creative"].items()
        }

        text_list = {
            "nominal": [t for t in dataset_mapping["nominal"]],
            "creative": [t for t in dataset_mapping["creative"]],
            "nominal-chain": [t for t in dataset_mapping["nominal-chain"]],
            "creative-chain": [t for t in dataset_mapping["creative-chain"]]
        }

        # Create an augmented version of the creative task
        # We want to ensure that the same test objects are used for "creative" and other prompts
        # Otherwise it will not be a fair comparison
        if mode == "creative-obj":
            dataset_mapping["creative-obj"] = {
                k: v for k, v in zip(augmented_prompts_obj, dataset_mapping["creative"].values())
            }
            text_list["creative-obj"] = [t for t in dataset_mapping["creative-obj"]]
        elif mode == "creative-task":
            dataset_mapping["creative-task"] = {
                k: v for k, v in zip(augmented_prompts_task, dataset_mapping["creative"].values())
            }
            text_list["creative-task"] = [t for t in dataset_mapping["creative-task"]]
        elif mode == "creative-task-obj":
            dataset_mapping["creative-task-obj"] = {
                k: v for k, v in zip(augmented_prompts_task_obj, dataset_mapping["creative"].values())
            }
            text_list["creative-task-obj"] = [t for t in dataset_mapping["creative-task-obj"]]
        elif mode == "creative-obj-chain":
            dataset_mapping["creative-obj-chain"] = {
                (k + " " + chain_of_thought): v for k, v in zip(augmented_prompts_obj, dataset_mapping["creative"].values())
            }
            text_list["creative-obj-chain"] = [t for t in dataset_mapping["creative-obj-chain"]]
        elif mode == "creative-task-chain":
            dataset_mapping["creative-task-chain"] = {
                (k + " " + chain_of_thought): v for k, v in zip(augmented_prompts_task, dataset_mapping["creative"].values())
            }
            text_list["creative-task-chain"] = [t for t in dataset_mapping["creative-task-chain"]]
        elif mode == "creative-task-obj-chain":
            dataset_mapping["creative-task-obj-chain"] = {
                (k + " " + chain_of_thought): v for k, v in zip(augmented_prompts_task_obj, dataset_mapping["creative"].values())
            }
            text_list["creative-task-obj-chain"] = [t for t in dataset_mapping["creative-task-obj-chain"]]

        assert len(text_list["nominal"]) == N_tasks

        is_cot_mode = "chain" in mode

        for text in text_list[mode]:
            images = []
            names = []
            for name, path in image_full_paths.items():
                if name in dataset_mapping[mode][text]:
                    images.append(Image.open(path))
                    names.append(name)

            if "vilt" in model_name:
                eval_output = run_vilt_eval(model, processor, text, images, names, device, return_scores=is_cot_mode)
            else:
                eval_output = run_clip_eval(model, processor, text, images, names, device, return_scores=is_cot_mode)

            if is_cot_mode:
                # Dynamically extract tool name based on the ground_truth nominal keys
                tool_name = next((tool for tool in ground_truth["nominal"].keys() if tool in text), "unknown tool")
                affordance_desc = tool_descriptions.get(tool_name, text)
                # Pass scores + context to LLM for CoT reasoning
                candidate_scores = eval_output
                predicted_object, reasoning = run_llm_cot(tool_name, affordance_desc, candidate_scores)
                
                if args.verbose:
                    print(f"Mode: {mode}, Text: {text}")
                    print(f"Vision Scores: {candidate_scores}")
                    print(f"LLM Reasoning:\n{reasoning}")
                    print(f"LLM Prediction: {predicted_object}\n")

                if args.save_reasoning:
                    # Log for later analysis
                    reasoning_log.append({
                        "text": text,
                        "vision_scores": candidate_scores,
                        "llm_selection": predicted_object,
                        "llm_reasoning": reasoning,
                        "correct": get_accuracy(text, predicted_object, ground_truth[mode])
                    })
            else:
                predicted_object = eval_output
                if args.verbose:
                    print(f"Mode: {mode}, Text: {text}, Object: {predicted_object}, All objects: {names}")
            
            accuracy += get_accuracy(text, predicted_object, ground_truth[mode])
            if text in accuracy_by_class:
                accuracy_by_class[text] += get_accuracy(text, predicted_object, ground_truth[mode])
            else:
                accuracy_by_class[text] = 1 #TODO: check if we should use `get_accuracy(text, predicted_object, ground_truth[mode])`

    if args.verbose:
        for k, v in accuracy_by_class.items():
            print(f"Accuracy for {k}: {v * 100/N_range}%")

    # Save reasoning log for qualitative analysis
    if args.save_reasoning and is_cot_mode:
        model_safe_name = model_name.split("/")[-1]
        log_file = f"reasoning_log_{mode}_{model_safe_name}.json"
        with open(log_file, "w") as f:
            json.dump(reasoning_log, f, indent=2)
        print(f"Reasoning log saved to {log_file}")
        
    # For visualization
    accuracy_by_class = {k: v / N_range for k, v in accuracy_by_class.items()}
    overall = np.mean(list(accuracy_by_class.values()))
    accuracy_by_class["overall"] = overall
    return accuracy_by_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description='Computational Creativity inspired prompting for creative problem solving',
        )
    parser.add_argument(
        "--task-type", type=str, required=True, help="Choose which prompt type to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Choose seed for experiment"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print results in console"
    )
    parser.add_argument(
        "--save-reasoning", action="store_true", help="Save LLM CoT reasoning to a JSON file"
    )
    args = parser.parse_args()
    assert args.task_type in [
        "creative",
        "nominal",
        "creative-obj",
        "creative-task",
        "creative-task-obj",
        "creative-chain",
        "nominal-chain",
        "creative-obj-chain",
        "creative-task-chain",
        "creative-task-obj-chain",
    ], "Allowed task types: creative/nominal/creative-obj/creative-task/creative-task-obj/creative-chain/nominal-chain/creative-obj-chain/creative-task-chain/creative-task-obj-chain"

    plotting_data = {}
    for name in hf_model_name.keys():
        print(f"Model: {name}")
        acc_by_class = main(hf_model_name[name], args)
        plotting_data[name] = acc_by_class

    print("Saving visualization...")
    plot_results(args.task_type, plotting_data)