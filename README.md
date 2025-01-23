# GSM8K-Ollama-Tester

This tool evaluates different prompting strategies for solving mathematical problems using LLMs through Ollama.

## Setup

1. Clone this repository:
```bash
git clone https://github.com/gitmichaelqiu/GSM8K-Ollama-Tester.git
cd GSM8K-Ollama-Tester
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Download the test data:
```bash
cd test_data
git clone https://github.com/openai/grade-school-math.git
```

## Configuration

### Setting the Model and Problem Count

The default settings are at the top of `evaluate_prompts.py`:

```python
MODEL = "your-model-name"
NUM_PROBLEMS = 10  # Number of problems to evaluate
RANDOM_SEED = 42
```

Modify these constants to change:
- The model used for evaluation
- The number of problems processed in each run
- The random seed for problem selection (problems are randomly sampled from the test set)

### Configuring Prompts

Prompts are defined in `prompts.json`. Each prompt should have a unique name and a template where `{problem}` will be replaced with the actual math problem.

## Running the Evaluation

1. Make sure your test data is in the correct location:
```
test_data/
└── grade-school-math/
    └── grade_school_math/
        └── data/
            └── test.jsonl
```

2. Run the evaluation script:
```bash
python evaluate_prompts.py
```

The script will randomly select the specified number of problems from the test set. The random selection is seeded for reproducibility.

## Output

The script creates a new directory for each run with the format `{model-name}_result-{number}`. Each directory contains:

1. `detailed_results.md`: Detailed analysis including:
   - Model name and number of problems evaluated
   - Each problem's question and correct answer
   - Model's response for each prompt
   - Whether each answer was correct

2. `results_summary.md`: A summary table showing:
   - Model name and number of problems evaluated
   - Success/failure for each prompt per problem
   - List of all prompts used

3. `prompt_comparison_test.jsonl.png`: Bar chart visualization showing:
   - Accuracy percentage for each prompt
   - Model name and number of problems evaluated
