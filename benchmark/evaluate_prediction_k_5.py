#!/usr/bin/env python3

import pandas as pd
import json
import pickle
import time
import os
from openai import OpenAI
from tqdm import tqdm
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_api():
    """Load API key and create OpenAI client"""
    with open("./config.json") as f:
        api_key = json.load(f)["api_key"]
        client = OpenAI(api_key=api_key)
    return client

class RecipeEvaluator:
    def __init__(self, model="gpt-4o-2024-05-13", temperature=0.1, max_tokens=4096):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = load_api()
        
        # Load evaluation prompt
        self.evaluation_prompt = """You are a materials synthesis evaluator.

You will receive:
- contribution: a brief summary of the key contribution of synthesis recipe
- predicted_recipe: target materials, used precursor, a step-by-step predicted synthesis procedure
- gt_recipe: the correct (ground truth) target materials, used precursor, a step-by-step synthesis procedure

---

Step 1: Determine scoring targets based on the process

Read the `contribution` and `predicted_recipe` to infer the synthesis process type and select the parameters to evaluate.

| Process                 | Parameters to extract and score       |
|------------------------|----------------------------------------|
| solid_state_reaction    | precursors, temperature, time          |
| mechanochemical         | precursors, rpm, time                  |
| sintering_consolidation | precursors, temperature, pressure      |
| thermal_pyrolysis       | precursors, temperature, time          |

---

Step 2: Parameter extraction and missing value handling

Precursors:
For both predicted and ground truth recipes, extract the precursors used in the initial synthesis step(s), typically in Step 1 and Step 2.
Ignore any materials that are only formed later (e.g., intermediates or coatings).
Do not rely solely on the "## Precursors" section—prioritize materials actually used in the initial few steps of the synthesis process.

Process parameters:
When extracting temperature, time, rpm, and pressure values, focus on the MAIN SYNTHESIS OF THE TARGET MATERIALS step. 
Exclude pre-treatment steps (drying, binder removal, low-temperature treatments) and post-treatment steps (stabilization, final annealing).

Missing value handling:
- If either predicted OR ground truth value is missing, mark as null (excluded from evaluation)
- If values use vague expressions without specific numbers ("sufficiently long", "high temperature", "appropriate pressure"), mark as null
- If comparison is impossible due to completely different methods (microwave vs conventional heating, manual vs mechanochemical), mark as null

---

Step 3: Unit conversion and comparability check

Unit standardization:
- Temperature: Convert all values to °C (Celsius)
- Time: Convert all values to h (hours) 
- Pressure: Convert all values to MPa
- RPM: Convert all values to rpm

Conversion guidelines:
- Perform standard conversions (°F→°C, min→h, GPa→MPa, etc.)
- If conversion is uncertain or impossible (e.g., Hz→rpm, acceleration→rpm), mark as null
- For range values, use middle value (e.g., "800-900°C" → 850°C)
- For multi-step processes, identify and use the main synthesis step value only

Comparability check:
- If processes use fundamentally different methods (manual grinding vs mechanochemical), mark as null
- If required parameter doesn't exist in one recipe (e.g., rpm evaluation but one uses pressureless process), mark as null

---

Step 4: Calculation and scoring (score: integer, max 5)

For each non-null parameter:

1. Extract predicted value (standardized): [number]
2. Extract GT value (standardized): [number]  
3. Calculate absolute error: |predicted - GT| = [result]
4. Match error to score table: [error range] → [score]
5. Assign final score: [score]

Scoring rules:

Precursor (chemical inputs)
| Score | Criteria |
|-------|-------------------------------------------------------------|
| 5     | Exactly same materials (precursors)                         |
| 4     | Different materials (precursors) but all target elements included |
| 3     | One element missing                                          |
| 2     | Two or more elements missing                                 |
| 1     | Largely different or unclear                                 |

Temperature (°C)
| Score | Absolute error |
|-------|----------------|
| 5     | ≤ 50 °C         |
| 4     | 51–100 °C       |
| 3     | 101–150 °C      |
| 2     | 151–200 °C      |
| 1     | > 200 °C        |

RPM
| Score | Absolute error |
|-------|----------------|
| 5     | ≤ 100 rpm       |
| 4     | 101–200 rpm     |
| 3     | 201–300 rpm     |
| 2     | 301–400 rpm     |
| 1     | > 400 rpm       |

Pressure (MPa)
| Score | Absolute error |
|-------|----------------|
| 5     | ≤ 20 MPa        |
| 4     | 21–40 MPa       |
| 3     | 41–60 MPa       |
| 2     | 61–80 MPa       |
| 1     | > 80 MPa        |

Time (h) — only penalize if predicted time is shorter than ground truth
Let t_gt be the ground truth time, and t_pred be the predicted time.

| Score | t_pred vs t_gt            |
|-------|---------------------------|
| 5     | t_pred ≥ t_gt             |
| 4     | t_gt − t_pred ≤ 3 h       |
| 3     | 3 < t_gt − t_pred ≤ 6 h   |
| 2     | 6 < t_gt − t_pred ≤ 9 h   |
| 1     | > 9 h shortage            |

---

Step 5: Final consistency check

Before providing your JSON output:

1. Review each parameter scoring:
   - Does your explanation match the assigned score?
   - Are your calculations consistent with the score tables?
   
2. Check for contradictions:
   - If you calculated "150°C error = 3 points" then JSON must show 3, not 1
   - If explanation says "missing value" then JSON must show null, not 1-5
   
3. Revise if needed:
   - If contradiction found, correct either explanation or score
   - Final explanation and JSON score must be identical
   - If invalid score detected, recalculate using score tables
   
4. Final verification:
   - All reasoning is logically coherent
   - All scores are within valid range (1-5 or null)
   - No self-contradictory statements remain

---

Step 6: Output format

Provide your scoring results in the following JSON format (no extra text):

{{
    "precursor_score": [integer 1-5 or null],
    "precursor_reason": "[brief explanation for precursor score or 'Missing value - excluded from evaluation']",
    "recipe_score": {{"parameter1": integer or null, "parameter2": integer or null}},
    "recipe_reason": {{"parameter1": "explanation or 'Missing value - excluded from evaluation'", "parameter2": "explanation or 'Missing value - excluded from evaluation'"}}
}}

Example for mechanochemical process:
{{
    "precursor_score": 4,
    "precursor_reason": "All required elements present but different compounds used",
    "recipe_score": {{"rpm": 3, "time": null}},
    "recipe_reason": {{"rpm": "Predicted 250 rpm vs GT 400 rpm, absolute error 150 rpm within 201-300 range", "time": "Missing value - excluded from evaluation"}}
}}

Include only the relevant process variables for the given synthesis type.
Use null for missing values instead of assigning arbitrary scores.
Ensure the output is valid JSON that can be directly parsed.

Here is the input data:

contribution:
{contribution}

predicted_recipe:
{predicted_recipe}

gt_recipe:
{gt_recipe}

"""

    def evaluate_recipe(self, contribution, predicted_recipe, gt_recipe, max_retries=5):
        """Evaluate recipe using LLM"""
        
        # Build prompt
        prompt = self.evaluation_prompt.format(
            contribution=contribution,
            predicted_recipe=predicted_recipe, 
            gt_recipe=gt_recipe
        )
        
        # Evaluate recipe with retries
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a materials synthesis evaluator. You must return ONLY a valid JSON object with the exact format specified. Do not include any explanatory text, markdown formatting, or code blocks."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                result = response.choices[0].message.content.strip()
                
                # Clean and parse JSON response
                try:
                    # Remove markdown code blocks if present
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0].strip()
                    elif "```" in result:
                        result = result.split("```")[1].split("```")[0].strip()
                    
                    # Try to find JSON-like content if response has extra text
                    if not result.strip().startswith("{"):
                        # Look for JSON pattern in the response
                        import re
                        json_match = re.search(r'\{.*\}', result, re.DOTALL)
                        if json_match:
                            result = json_match.group(0)
                        else:
                            raise json.JSONDecodeError("No JSON found", result, 0)
                    
                    parsed_result = json.loads(result)
                    return parsed_result
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error (attempt {attempt + 1}): {e}")
                    print(f"Raw response: {result[:200]}...")
                    if attempt < max_retries - 1:
                        print(f"Retrying...")
                        time.sleep(1)
                        continue
                    else:
                        return "error"
                
            except Exception as e:
                print(f"API error evaluating recipe (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return "error"
        
        return "error"

def process_single_row(args):
    """Process a single row - evaluate predicted recipe against ground truth"""
    evaluator, idx, process, predicted_recipe, gt_recipe = args
    try:
        if pd.isna(predicted_recipe) or pd.isna(gt_recipe) or predicted_recipe == "error" or gt_recipe == "error":
            return idx, "error", "Missing or invalid recipe data", "error", "Missing or invalid recipe data"
        
        evaluation_result = evaluator.evaluate_recipe(process, predicted_recipe, gt_recipe)
        time.sleep(0.5)  # Rate limiting
        
        if evaluation_result == "error":
            return idx, "error", "Evaluation failed", "error", "Evaluation failed"
        
        # Extract results from JSON with error handling
        try:
            if isinstance(evaluation_result, dict):
                precursor_score = evaluation_result.get("precursor_score", "error")
                precursor_reason = evaluation_result.get("precursor_reason", "Evaluation failed")
                recipe_score = evaluation_result.get("recipe_score", "error")
                recipe_reason = evaluation_result.get("recipe_reason", "Evaluation failed")
            else:
                return idx, "error", "Invalid evaluation result format", "error", "Invalid evaluation result format"
        except Exception as e:
            return idx, "error", f"Result parsing error: {e}", "error", f"Result parsing error: {e}"
        
        return idx, precursor_score, precursor_reason, recipe_score, recipe_reason
        
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        return idx, "error", f"Processing error: {e}", "error", f"Processing error: {e}"

def process_batch(df_batch, batch_id, evaluator, max_workers=3):
    """Process a batch of rows using thread pool"""
    print(f"Processing batch {batch_id} with {len(df_batch)} rows...")
    
    # Prepare arguments for each row
    tasks = []
    for idx, row in df_batch.iterrows():
        # Skip if already processed
        if pd.notna(row.get("precursor_score")):
            continue
            
        tasks.append((evaluator, idx, row["process"], row["predicted_recipe"], row["gt_recipe"]))
    
    if not tasks:
        print(f"Batch {batch_id}: All rows already processed")
        return df_batch
    
    # Process with thread pool
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(process_single_row, task): task for task in tasks}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(tasks), desc=f"Batch {batch_id}", leave=False) as pbar:
            for future in as_completed(future_to_task):
                try:
                    idx, precursor_score, precursor_reason, recipe_score, recipe_reason = future.result()
                    results[idx] = {
                        "precursor_score": precursor_score,
                        "precursor_reason": precursor_reason,
                        "recipe_score": recipe_score,
                        "recipe_reason": recipe_reason
                    }
                    pbar.update(1)
                except Exception as e:
                    task = future_to_task[future]
                    idx = task[1]
                    results[idx] = {
                        "precursor_score": "error",
                        "precursor_reason": f"Task failed: {e}",
                        "recipe_score": "error",
                        "recipe_reason": f"Task failed: {e}"
                    }
                    print(f"Row {idx} failed: {e}")
                    pbar.update(1)
    
    # Update dataframe with results - use .at for single value assignment
    for idx, result in results.items():
        try:
            df_batch.at[idx, "precursor_score"] = result["precursor_score"]
            df_batch.at[idx, "precursor_reason"] = result["precursor_reason"]
            df_batch.at[idx, "recipe_score"] = result["recipe_score"]
            df_batch.at[idx, "recipe_reason"] = result["recipe_reason"]
        except Exception as e:
            print(f"Error updating row {idx}: {e}")
    
    print(f"Batch {batch_id} completed: {len(results)} rows processed")
    return df_batch

def save_progress(df, filepath_base):
    """Save progress to Pickle"""
    output_dir = "benchmark_result_k_5"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        pkl_path = os.path.join(output_dir, f"{filepath_base}.pkl")
        with open(pkl_path, "wb") as fw:
            pickle.dump(df, fw)
        print(f"Saved: {pkl_path}")
        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False

def find_last_progress(batch_size=50):
    """Find last completed batch and return start index"""
    output_dir = "benchmark_result_k_5"
    
    if not os.path.exists(output_dir):
        return 0, None
    
    # Find all batch files
    batch_files = []
    for filename in os.listdir(output_dir):
        if filename.startswith("evaluation_batch_") and filename.endswith(".pkl"):
            try:
                batch_num_str = filename.replace("evaluation_batch_", "").replace(".pkl", "")
                batch_num = int(batch_num_str)
                batch_files.append((batch_num, filename))
            except ValueError:
                continue
    
    if not batch_files:
        return 0, None
    
    # Get latest batch
    batch_files.sort(key=lambda x: x[0])
    latest_batch_num, latest_filename = batch_files[-1]
    
    try:
        latest_file_path = os.path.join(output_dir, latest_filename)
        with open(latest_file_path, 'rb') as f:
            df_saved = pickle.load(f)
        
        # Find processed rows
        processed = df_saved[df_saved["precursor_score"].notna()]
        if len(processed) == 0:
            return 0, None
        
        last_processed_idx = processed.index.max()
        current_batch = last_processed_idx // batch_size
        batch_start = current_batch * batch_size
        batch_end = min(batch_start + batch_size, len(df_saved))
        
        # Check if current batch is complete
        current_batch_df = df_saved.loc[batch_start:batch_end-1]
        current_batch_processed = len(current_batch_df[current_batch_df["precursor_score"].notna()])
        expected_in_batch = batch_end - batch_start
        
        if current_batch_processed == expected_in_batch:
            print(f"Batch {current_batch + 1} complete, starting from batch {current_batch + 2}")
            return batch_end, df_saved
        else:
            print(f"Batch {current_batch + 1} incomplete, restarting from batch {current_batch + 1}")
            return batch_start, df_saved
            
    except Exception as e:
        print(f"Error loading progress: {e}")
        return 0, None

def evaluate_recipes(
    input_file="./dataset_predict_n_gt_k_5.pkl",
    model="gpt-4.1-2025-04-14", 
    batch_size=50,
    max_workers=3
):
    """Evaluate predicted recipes against ground truth for all entries in DataFrame"""
    
    # Find last progress
    start_idx, df_saved = find_last_progress(batch_size)
    
    if df_saved is not None:
        df = df_saved
        print("Loaded previous progress")
    else:
        # Load original data
        with open(input_file, "rb") as f:
            df = pickle.load(f)
        print(f"Loaded original data: {len(df)} rows")
    
    # Create evaluation columns if they don't exist
    for col in ["precursor_score", "precursor_reason", "recipe_score", "recipe_reason"]:
        if col not in df.columns:
            df[col] = None
    
    # Initialize recipe evaluator
    print(f"Initializing recipe evaluator (model: {model})...")
    evaluator = RecipeEvaluator(model=model)
    
    # Filter data that needs processing
    total_rows = len(df)
    print(f"Starting from index {start_idx} out of {total_rows} total rows")
    
    remaining_df = df.iloc[start_idx:].copy()
    num_batches = (len(remaining_df) + batch_size - 1) // batch_size
    print(f"Will process {num_batches} batches")
    
    try:
        for batch_idx in range(num_batches):
            start_row = batch_idx * batch_size
            end_row = min(start_row + batch_size, len(remaining_df))
            
            actual_batch_num = (start_idx // batch_size) + batch_idx + 1
            
            # Get batch with proper index alignment
            batch_start_idx = start_idx + start_row
            batch_end_idx = start_idx + end_row
            batch_df = df.iloc[batch_start_idx:batch_end_idx].copy()
            
            print(f"\n=== Batch {actual_batch_num} ({batch_idx + 1}/{num_batches}) ===")
            print(f"Processing rows {start_idx + start_row} to {start_idx + end_row - 1}")
            
            # Process batch
            processed_batch = process_batch(batch_df, actual_batch_num, evaluator, max_workers)
            
            # Update main dataframe - use direct assignment instead of update
            for idx, row in processed_batch.iterrows():
                if pd.notna(row["precursor_score"]):  # Only update if processed
                    df.at[idx, "precursor_score"] = row["precursor_score"]
                    df.at[idx, "precursor_reason"] = row["precursor_reason"] 
                    df.at[idx, "recipe_score"] = row["recipe_score"]
                    df.at[idx, "recipe_reason"] = row["recipe_reason"]
            
            # Save progress
            save_filepath = f"evaluation_batch_{actual_batch_num:03d}"
            save_progress(df, save_filepath)
            
            # Show progress
            current_processed = len(df[df["precursor_score"].notna()])
            print(f"Progress: {current_processed}/{total_rows} evaluations completed")
            
            time.sleep(1)  # Brief pause between batches
    
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")
        save_progress(df, "evaluation_interrupted")
        return df
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        save_progress(df, "evaluation_error")
        return df
    
    # Final save
    print("\nEvaluation complete! Saving final results...")
    save_progress(df, "evaluation_final")
    
    # Summary
    print("\n=== Evaluation Results Summary ===")
    completed = len(df[df["precursor_score"].notna()])
    errors = len(df[df["precursor_score"] == "error"])
    print(f"Total evaluations completed: {completed}/{total_rows}")
    print(f"Errors: {errors}")
    
    return df

if __name__ == "__main__":
    # Create output directory
    output_dir = "0910_benchmark_result_k_5"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved to: {os.path.abspath(output_dir)}/")
    
    df = evaluate_recipes(
        model="gpt-4.1-2025-04-14",
        batch_size=50,
        max_workers=3
    )
    print("Recipe evaluation completed!")
    print(f"All results saved in: {os.path.abspath(output_dir)}/")
