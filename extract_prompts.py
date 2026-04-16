import json
import os

INSTRUCTIONS = """# User Simulation Guidelines
You are playing the role of a customer contacting a customer service representative. 
Your goal is to simulate realistic customer interactions while following specific scenario instructions.

## Core Principles
- Generate one message at a time, maintaining natural conversation flow.
- Strictly follow the scenario instructions you have received.
- Never make up or hallucinate information not provided in the scenario instructions. Information that is not provided in the scenario instructions should be considered unknown or unavailable.
- Avoid repeating the exact instructions verbatim. Use paraphrasing and natural language to convey the same information
- Disclose information progressively. Wait for the agent to ask for specific information before providing it.

## Task Completion
- The goal is to continue the conversation until the task is complete.
- If the instruction goal is satisified, generate the '###STOP###' token to end the conversation.
- If you are transferred to another agent, generate the '###TRANSFER###' token to indicate the transfer.
- If you find yourself in a situation in which the scenario does not provide enough information for you to continue the conversation, generate the '###OUT-OF-SCOPE###' token to end the conversation.
Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions and maintaining character consistency."""


domain = "retail"
task_path = os.path.join(os.getcwd(), "data", "tau2", "domains", domain, "tasks.json")

# Read the json file

with open(task_path, "r") as f:
    tasks = json.load(f)

prompts_data = []

for task in tasks:
    id = task["id"]
    purpose = task["description"]["purpose"]
    task_instructions = task["user_scenario"]["instructions"]["task_instructions"]
    reason_for_call = task["user_scenario"]["instructions"]["reason_for_call"]
    known_info = task["user_scenario"]["instructions"]["known_info"]
    evaluation_criteria_nl_assertions = task["evaluation_criteria"]["nl_assertions"]

    input_prompt = f"""{INSTRUCTIONS}

<scenario>
Instructions:
    Domain: {domain}
    Reason for call:
        {reason_for_call}
    Known Info:
        {known_info}
    Task Instructions:
        {task_instructions}
</scenario>
"""
    
    prompts_data.append({
        "id": id,
        "input_prompt": input_prompt,
        "evaluation_criteria_nl_assertions": evaluation_criteria_nl_assertions,
        "purpose": purpose,
        "task_instructions": task_instructions,
        "reason_for_call": reason_for_call,
        "known_info": known_info
    })

dataset_path = os.path.join(os.getcwd(), "dataset.json")

with open(dataset_path, "r") as f:
    dataset = json.load(f)
dataset[f"tau2-bench-domain-{domain}"] = prompts_data

with open(dataset_path, "w") as f:
    json.dump(dataset, f, indent=4)