import json
import os
import requests
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import glob
import random

MODEL = "qwen2.5:0.5b"
NUM_PROBLEMS = 50  # Constant for number of problems to evaluate
RANDOM_SEED = 42  # For reproducibility

def load_prompts(prompt_file):
    """Load prompts from a JSON file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)

def get_model_response(prompt, problem, model=MODEL):
    """Get response from Ollama model."""
    url = "http://127.0.0.1:11434/api/generate"
    
    # Combine the prompt template with the actual problem
    full_prompt = prompt.replace("{problem}", problem)
    
    data = {
        "model": model,
        "prompt": full_prompt,
        "stream": False
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return None

def extract_answer(response):
    """Extract the final answer from the model's response."""
    try:
        # Look for the last number in the response
        words = response.split()
        for word in reversed(words):
            # Clean the word of any non-numeric characters except decimal points and negative signs
            cleaned = ''.join(c for c in word if c.isdigit() or c in '.-')
            if cleaned and cleaned not in ['.', '-']:
                return float(cleaned)
    except:
        return None
    return None

def load_jsonl(file_path):
    """Load problems from a JSONL file."""
    problems = []
    with open(file_path, 'r') as f:
        for line in f:
            problems.append(json.loads(line.strip()))
    return problems

def extract_grade_school_answer(answer_str):
    """Extract the final numerical answer from grade-school-math format."""
    try:
        # The answer is typically the last number in the string
        # It's often after the last '=' sign or at the very end
        parts = answer_str.split('=')
        last_part = parts[-1].strip()
        
        # Extract the last number from the string
        words = last_part.split()
        for word in reversed(words):
            # Clean the word of any non-numeric characters except decimal points and negative signs
            cleaned = ''.join(c for c in word if c.isdigit() or c in '.-')
            if cleaned and cleaned not in ['.', '-']:
                return float(cleaned)
    except:
        return None
    return None

def generate_detailed_report(problems_data, filename, model_name):
    """Generate a detailed report of all problems and answers."""
    with open(filename, 'w') as f:
        f.write("# Detailed Evaluation Report\n\n")
        f.write(f"## Model: {model_name}\n")
        f.write(f"## Number of Problems: {NUM_PROBLEMS}\n\n")
        for idx, data in enumerate(problems_data, 1):
            f.write(f"## Problem {idx}\n\n")
            f.write(f"### Question:\n")
            f.write(f"{data['question']}\n\n")
            f.write(f"### Correct Answer:\n")
            f.write(f"{data['correct_answer']}\n\n")
            f.write(f"### Model Answers:\n")
            for prompt_name, response in data['responses'].items():
                f.write(f"#### {prompt_name}\n")
                f.write(f"##### Full Response\n")
                f.write(f"{response['full_response']}\n")
                f.write(f"##### Extracted Answer\n")
                f.write(f"{response['extracted_answer']}\n")
                f.write(f"##### Correct: {'✓' if response['is_correct'] else '✗'}\n")
            f.write("---\n\n")

def generate_markdown_table(problems_data, filename, prompts, model_name):
    """Generate a markdown table with problem results."""
    with open(filename, 'w') as f:
        # Write header with model info
        f.write("# Evaluation Results Summary\n\n")
        f.write(f"## Model: {model_name}\n")
        f.write(f"## Number of Problems: {NUM_PROBLEMS}\n\n")
        f.write("| Problem # |")
        prompt_names = list(problems_data[0]['responses'].keys())
        for prompt in prompt_names:
            f.write(f" {prompt} |")
        f.write("\n|")
        f.write(" --- |" * (len(prompt_names) + 1))
        f.write("\n")
        
        # Write rows
        for idx, data in enumerate(problems_data, 1):
            f.write(f"| {idx} |")
            for prompt in prompt_names:
                result = data['responses'][prompt]['is_correct']
                f.write(f" {'✓' if result else '✗'} |")
            f.write("\n")

        # Write prompts
        f.write("\n## Prompts\n\n")
        for prompt in prompt_names:
            f.write(f"**{prompt}**\n\n")
            f.write(f"{prompts[prompt]}\n\n")

def get_next_result_number(model_name):
    """Get the next result number for the given model."""
    # List all directories that match the pattern model_name_result-*
    existing_dirs = glob.glob(f"{model_name}_result-*")
    if not existing_dirs:
        return 1
    
    # Extract numbers from directory names
    numbers = []
    for dir_name in existing_dirs:
        try:
            # Extract number after "result-"
            num = int(dir_name.split('result-')[1])
            numbers.append(num)
        except:
            continue
    
    # Return the next number
    return max(numbers, default=0) + 1

def create_result_directory(model_name):
    """Create a new result directory with incrementing number."""
    # Replace invalid characters in model name
    safe_model_name = model_name.replace('/', '-').replace(':', '-').replace('\\', '-')
    
    next_num = get_next_result_number(safe_model_name)
    result_dir = f"{safe_model_name}_result-{next_num}"
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def evaluate_test_set(test_file, prompts, model=MODEL):
    """Evaluate a single test file using all prompts."""
    # Create result directory
    result_dir = create_result_directory(model)
    
    results = {prompt_name: {'correct': 0, 'total': 0} for prompt_name in prompts}
    problems_data = []
    
    # Load problems from JSONL file
    problems = load_jsonl(test_file)
    
    # Randomly select NUM_PROBLEMS problems
    random.seed(RANDOM_SEED)  # Set seed for reproducibility
    selected_problems = random.sample(problems, NUM_PROBLEMS)
    
    for problem in tqdm(selected_problems, desc=f"Processing {os.path.basename(test_file)}"):
        question = problem.get('question', '')
        correct_answer = extract_grade_school_answer(problem.get('answer', '0'))
        
        problem_data = {
            'question': question,
            'correct_answer': correct_answer,
            'responses': {}
        }
        
        if correct_answer is not None:
            for prompt_name, prompt_template in prompts.items():
                response = get_model_response(prompt_template, question, model)
                if response:
                    predicted_answer = extract_answer(response)
                    is_correct = False
                    if predicted_answer is not None:
                        results[prompt_name]['total'] += 1
                        is_correct = abs(predicted_answer - correct_answer) < 1e-6
                        if is_correct:
                            results[prompt_name]['correct'] += 1
                    
                    problem_data['responses'][prompt_name] = {
                        'full_response': response,
                        'extracted_answer': predicted_answer,
                        'is_correct': is_correct
                    }
        
        problems_data.append(problem_data)
    
    # Generate reports in the result directory with model information
    generate_detailed_report(problems_data, os.path.join(result_dir, 'detailed_results.md'), model)
    generate_markdown_table(problems_data, os.path.join(result_dir, 'results_summary.md'), prompts, model)
    
    return results, result_dir

def visualize_results(results, test_name, result_dir, model_name):
    """Create a bar chart showing the accuracy for each prompt."""
    prompt_names = list(results.keys())
    accuracies = [results[name]['correct'] / results[name]['total'] * 100 if results[name]['total'] > 0 else 0 
                 for name in prompt_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(prompt_names, accuracies)
    plt.title(f'Prompt Performance Comparison\nModel: {model_name} - {test_name}\nNumber of Problems: {NUM_PROBLEMS}')
    plt.xlabel('Prompts')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f'prompt_comparison_{os.path.basename(test_name)}.png'))
    plt.close()

def main():
    # Load prompts from JSON file
    prompt_file = "prompts.json"
    prompts = load_prompts(prompt_file)
    
    # Process only grade-school-math test set
    test_file = 'test_data/grade-school-math/grade_school_math/data/test.jsonl'
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
    
    print(f"\nProcessing test set: {test_file}")
    print(f"Using model: {MODEL}")
    print(f"Number of problems: {NUM_PROBLEMS}")
    results, result_dir = evaluate_test_set(test_file, prompts, MODEL)
    
    # Print results
    print("\nResults:")
    for prompt_name, result in results.items():
        accuracy = (result['correct'] / result['total'] * 100) if result['total'] > 0 else 0
        print(f"{prompt_name}: {accuracy:.2f}% ({result['correct']}/{result['total']})")
    
    # Visualize results with model information
    visualize_results(results, os.path.basename(test_file), result_dir, MODEL)
    
    print(f"\nResults saved in directory: {result_dir}")

if __name__ == "__main__":
    main() 