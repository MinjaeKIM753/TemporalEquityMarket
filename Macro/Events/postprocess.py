import json
import os

# Updates the main batch aggregation file with results from a recollection batch.
# This is used to correct entries in BatchAggreg.jsonl for dates that were initially
# processed incorrectly and listed in recollect.jsonl, then re-processed via LLM.
def update_batch_aggreg():
    batch_recollect_file = "Macro/Events/BatchRecollect_tasks_output.jsonl"
    batch_aggreg_file = "Macro/Events/BatchAggreg.jsonl"

    if not os.path.exists(batch_recollect_file):
        print(f"Error: Recollection file not found: {batch_recollect_file}")
        return
    if not os.path.exists(batch_aggreg_file):
        print(f"Error: Aggregation file not found: {batch_aggreg_file}")
        return

    print(f"Loading data from {batch_recollect_file} and {batch_aggreg_file}")
    try:
        with open(batch_recollect_file, "r", encoding="utf-8") as f:
            batch_recollect_tasks = [json.loads(line) for line in f]
        with open(batch_aggreg_file, "r", encoding="utf-8") as f:
            batch_aggreg_tasks = [json.loads(line) for line in f]
    except json.JSONDecodeError as e:
        print(f"Error reading JSONL file: {e}")
        return
    except Exception as e:
        print(f"Error opening or reading files: {e}")
        return

    print("Creating dictionary from recollection tasks...")
    recollect_dict = {}
    for task in batch_recollect_tasks:
        custom_id = task.get("custom_id")
        if custom_id:
            recollect_dict[custom_id] = task
        else:
            print(f"Warning: Task missing 'custom_id' in {batch_recollect_file}")

    updated_tasks = []
    update_count = 0
    print("Processing and updating tasks...")
    for task in batch_aggreg_tasks:
        custom_id = task.get("custom_id")
        if custom_id in recollect_dict:
            updated_tasks.append(recollect_dict[custom_id])
            update_count += 1
        else:
            updated_tasks.append(task)

    print(f"Writing updated tasks back to {batch_aggreg_file}...")
    try:
        with open(batch_aggreg_file, "w", encoding="utf-8") as f:
            for task in updated_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing updated file: {e}")
        return

    print(f"Update complete. {update_count} tasks updated in {batch_aggreg_file}.")

if __name__ == "__main__":
    update_batch_aggreg()
