import os
import json
import time
import sys
from datetime import date, timedelta
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Access the API key
api_key_path = "Macro/Events/api_key.txt"
if not os.path.exists(api_key_path):
    raise FileNotFoundError(f"{api_key_path} not found.")
with open(api_key_path, "r") as f:
    api_key = f.read().strip()
client = OpenAI(api_key=api_key)

# System prompt for event generation task
SYSTEM_MESSAGE = (
    '''<SYSTEM_PROMPT>
    <instructions>
    <role_note>You are a historical news retrieval model with extensive internal knowledge of world events.</role_note>
    <objective>For the target date provided by the user, list 18 significant events divided into three sections: 3 international events; for each of the following countries ["Hong Kong", "South Korea", "U.S", "Japan"], list 3 national events (total 12); and 3 events related to scientific, industrial, or social/technology news. Emphasize objective, fact-based reporting.</objective>
    <reference_date>Use the date provided in the user message as the target. If no events are found for that date, check the previous day; if still insufficient, check 2 days before, and finally 3 days before. Do not check further than 3 days prior.</reference_date>
    <hidden_thinking>(Think step-by-step to recall actual historical events from your training data for the target date or its fallback dates.)</hidden_thinking>
    <guidelines>
      <complete>Ensure your answer is complete and not truncated by token limits.</complete>
      <categories>
        <global>
          <count>3</count>
          <description>List the top 3 events that affected the world globally on (or near) the target date.</description>
        </global>
        <national>
          <count>12</count>
          <description>For each of the following countries: "Hong Kong", "South Korea", "U.S", and "Japan", list the top 3 events that affected the country on (or near) the target date, with an emphasis on economic and national developments.</description>
        </national>
        <news>
          <count>3</count>
          <description>List the top 3 events related to scientific, industrial, or social/technology news on (or near) the target date.</description>
        </news>
      </categories>
      <fallback>
        <recheck>If no events are available for the target date in any category, recheck events from the previous day; if still insufficient, then from 2 days before, and finally from 3 days before.</recheck>
        <none>Only return 'None' for a category if, after checking the target date and the preceding 3 days, you are absolutely certain that no event occurred.</none>
        <verification>Always double-check for any valid events using your internal historical knowledge before resorting to 'None'.</verification>
      </fallback>
      <special_note>If the target date is the first working day of the week (following a weekend or holiday), include significant events from the non-working days since the last working day as well.</special_note>
    </guidelines>
    <format>
      <schema>
        {
          "type": "array",
          "items": {
              "type": "object",
              "properties": {
                  "Category": {
                      "type": "string",
                      "enum": ["International", "National", "Social/Technology"],
                      "description": "The event category. For national events, include a 'Country' field as well."
                  },
                  "Title": {
                      "type": "string",
                      "description": "A concise and objective event title based on verifiable historical facts; if no event exists, write 'None'."
                  },
                  "Description": {
                      "type": "string",
                      "description": "A detailed description of the event in exactly 6 sentences, covering what happened, its consequences, and its implications. If no event exists, write 'None'. Ensure the description is objective, fact-based, and grounded in verifiable historical records."
                  },
                  "Country": {
                      "type": "string",
                      "enum": ["Hong Kong", "South Korea", "U.S", "Japan"],
                      "description": "For national events, specify the country. Omit this field for non-national events."
                  }
              },
              "required": ["Category", "Title", "Description"],
              "if": {
                  "properties": { "Category": { "const": "National" } }
              },
              "then": {
                  "required": ["Country"]
              },
              "additionalProperties": false
          },
          "minItems": 18,
          "maxItems": 18
        }
      </schema>
    </format>
    <output>Return only the JSON object that strictly conforms to the schema above with no additional text.</output>
  </instructions>
</SYSTEM_PROMPT>
'''
)

def generate_tasks(start: date, end: date):
    tasks = []
    current = start
    while current <= end:
        if current.weekday() < 5: 
            custom_id = f"Event_{current.isoformat()}"
            task = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": CFG.MODEL_NAME,
                    "max_tokens": CFG.MAX_COMPLETION_TOKENS,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MESSAGE}, 
                        {"role": "user", "content": f"Investigate events on {current.isoformat()}."}
                    ],
                    "response_format": {"type": "json_object"}, 
                    "temperature": 0.2 
                }
            }
            tasks.append(task)
        current += timedelta(days=1)
    return tasks

def process_batch(batch_name: str, start: date, end: date):
    tasks = generate_tasks(start, end)
    total_tasks = len(tasks)
    tasks_filename = f"Macro/Events/Saves/{batch_name}_tasks.jsonl"
    os.makedirs(os.path.dirname(tasks_filename), exist_ok=True)
    with open(tasks_filename, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"[{batch_name}] Wrote {total_tasks} tasks to {tasks_filename}")

    print(f"[{batch_name}] Uploading batch input file...")
    with open(tasks_filename, "rb") as file_handle:
        batch_file = client.files.create(
            file=file_handle,
            purpose="batch"
        )
    print(f"[{batch_name}] Uploaded file. File ID: {batch_file.id}")

    print(f"[{batch_name}] Creating batch job...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Batch job for events from {start.isoformat()} to {end.isoformat()}"
        }
    )
    print(f"[{batch_name}] Batch job created. Job ID: {batch_job.id}")
    print(f"[{batch_name}] Monitoring batch job {batch_job.id}...")

    while True:
        try:
            current_job = client.batches.retrieve(batch_job.id)
            status = current_job.status
            completed = current_job.request_counts.completed
            total = current_job.request_counts.total
            status_message = f"[{batch_name}] Status: {status} - Completed: {completed}/{total}"
            print("\r" + status_message + " " * 10, end="")

            if status in ["completed", "failed", "expired", "cancelled"]:
                break
        except Exception as e:
            print(f"\nError retrieving batch status: {e}")
            if "not found" in str(e).lower(): 
                print("Batch job not found, exiting monitoring.")
                return 
            time.sleep(30) 
            continue

        time.sleep(20)

    print(f"\n[{batch_name}] Final batch status: {status}")

    if status == "completed" and current_job.output_file_id:
        print(f"[{batch_name}] Retrieving batch output file...")
        try:
            output_file_content = client.files.content(current_job.output_file_id)
            output_filename = f"Macro/Events/Processed_files/{batch_name}_output.jsonl"
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            with open(output_filename, "wb") as f: # Write in binary mode
                f.write(output_file_content.read())
            print(f"[{batch_name}] Saved output file to {output_filename}")

        except Exception as e:
            print(f"\n[{batch_name}] Error retrieving or saving output file: {e}")

    elif current_job.error_file_id:
         print(f"[{batch_name}] Batch completed with errors. Error file ID: {current_job.error_file_id}")
    else:
        print(f"[{batch_name}] Batch did not complete successfully or has no output file.")

# For single date processing (direct API call)
def process_single_date(date_obj: date):
    try:
        response = client.chat.completions.create(
            model=CFG.MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_MESSAGE}, # Changed role to system
                {"role": "user", "content": f"Investigate events on {date_obj.isoformat()}."}
            ],
            max_tokens=CFG.MAX_COMPLETION_TOKENS, # Adjusted parameter name
            response_format={"type": "json_object"}, # Enforce JSON output
            temperature=0.2 # Lower temperature for fact-based generation
        )

        content = response.choices[0].message.content
        try:
            events_data = json.loads(content)
        except json.JSONDecodeError as json_err:
            print(f"\nError decoding JSON for {date_obj.isoformat()}: {json_err}")
            print(f"Raw content: {content[:500]}...") # Log partial raw content
            return {
                "date": date_obj.isoformat(),
                "error": f"JSONDecodeError: {json_err}",
                "raw_content": content
            }

        return {
            "date": date_obj.isoformat(),
            "events": events_data
        }
    except Exception as e:
        print(f"\nError processing {date_obj.isoformat()}: {type(e).__name__} - {str(e)}")
        return {
            "date": date_obj.isoformat(),
            "error": f"{type(e).__name__}: {str(e)}"
        }

def process_direct_dates(batch_name: str, dates: list, max_workers=2):
    total_dates = len(dates)
    print(f"[{batch_name}] Processing {total_dates} specific dates directly (max_workers={max_workers})")
    results = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_date = {executor.submit(process_single_date, d): d for d in dates}
        for future in as_completed(future_to_date):
            date_obj = future_to_date[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"\nError in future execution for {date_obj.isoformat()}: {str(e)}")
                results.append({
                    "date": date_obj.isoformat(),
                    "error": f"Future execution error: {str(e)}"
                })
            completed_count += 1
            progress = (completed_count / total_dates) * 100
            status_message = f"[{batch_name}] Progress: {completed_count}/{total_dates} ({progress:.1f}%)"
            print("\r" + status_message + " " * 10, end="") # Add padding

    print(f"\n[{batch_name}] Direct processing completed")

    # Ensure results are sorted by date
    results.sort(key=lambda x: x.get('date', '0000-00-00'))

    output_filename = f"Macro/Events/Processed_files/{batch_name}_direct_output.jsonl"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    with open(output_filename, "w", encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[{batch_name}] Direct results saved to {output_filename}")
    return results

def process_batch_dates(batch_name: str, dates: list):
    tasks = []
    for date_obj in dates:
        custom_id = f"Event_{date_obj.isoformat()}" # Consistent custom_id prefix
        task = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": CFG.MODEL_NAME,
                "max_tokens": CFG.MAX_COMPLETION_TOKENS, # Adjusted parameter name
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE}, # Changed role to system
                    {"role": "user", "content": f"Investigate events on {date_obj.isoformat()}."}
                ],
                "response_format": {"type": "json_object"}, # Enforce JSON output
                "temperature": 0.2 # Lower temperature for fact-based generation
            }
        }
        tasks.append(task)

    total_tasks = len(tasks)
    tasks_filename = f"Macro/Events/Saves/{batch_name}_tasks.jsonl" # Save tasks file
    os.makedirs(os.path.dirname(tasks_filename), exist_ok=True)

    with open(tasks_filename, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"[{batch_name}] Wrote {total_tasks} tasks for specific dates to {tasks_filename}")

    print(f"[{batch_name}] Uploading batch input file for specific dates...")
    with open(tasks_filename, "rb") as file_handle:
        batch_file = client.files.create(
            file=file_handle,
            purpose="batch"
        )
    print(f"[{batch_name}] Uploaded file. File ID: {batch_file.id}")

    print(f"[{batch_name}] Creating batch job for specific dates...")
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Batch job for specific events dates: {batch_name}"
        }
    )
    print(f"[{batch_name}] Batch job created. Job ID: {batch_job.id}")
    print(f"[{batch_name}] Monitoring batch job {batch_job.id}...")

    # Identical monitoring loop as in process_batch
    while True:
        try:
            current_job = client.batches.retrieve(batch_job.id)
            status = current_job.status
            completed = current_job.request_counts.completed
            total = current_job.request_counts.total
            status_message = f"[{batch_name}] Status: {status} - Completed: {completed}/{total}"
            print("\r" + status_message + " " * 10, end="") # Add padding

            if status in ["completed", "failed", "expired", "cancelled"]:
                break
        except Exception as e:
            print(f"\nError retrieving batch status: {e}")
            if "not found" in str(e).lower():
                print("Batch job not found, exiting monitoring.")
                return # Exit if batch not found
            time.sleep(30)
            continue
        time.sleep(20)

    print(f"\n[{batch_name}] Final batch status: {status}")

    if status == "completed" and current_job.output_file_id:
        print(f"[{batch_name}] Retrieving batch output file...")
        try:
            output_file_content = client.files.content(current_job.output_file_id)
            output_filename = f"Macro/Events/Processed_files/{batch_name}_output.jsonl"
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            with open(output_filename, "wb") as f: # Write in binary mode
                f.write(output_file_content.read())
            print(f"[{batch_name}] Saved output file to {output_filename}")
        except Exception as e:
             print(f"\n[{batch_name}] Error retrieving or saving output file: {e}")
    elif current_job.error_file_id:
         print(f"[{batch_name}] Batch completed with errors. Error file ID: {current_job.error_file_id}")
    else:
        print(f"[{batch_name}] Batch did not complete successfully or has no output file.")

class CFG:
    MODEL_NAME = "o3-mini-2025-01-31"
    MAX_COMPLETION_TOKENS = 8500 # 8000 fails over
    REASONING_EFFORT = 'medium' # high reasoning will cause more token consumption and hallucination
    

if __name__ == "__main__":
    # From Early 2025, OpenAI announced that sharing agreement of prompts will grant free of charge up to 11M tokens daily.
    # use dates adequately to consume tokens below 11M (Empirically about 5 years)
    START_DATE = date(1990, 1, 1)
    END_DATE = date(1994, 12, 31)
    MAX_WORKERS = 2 # Running with 2 workers 
    USE_BATCH = True
    HALF_YEAR = True # Split by half
    RECOLLECT_MODE = True # Recollect mode to run against the data that are not collected correctly
    
    if RECOLLECT_MODE:
        recollect_dates = []
        with open("Macro/Events/recollect.jsonl", "r") as f:
            for line in f:
                date_str = line.strip()
                recollect_dates.append(date.fromisoformat(date_str))
        process_batch_dates(
            "BatchRecollect",
            recollect_dates
        )
    else: # Collecting events by START_DATE and END_DATE
        if HALF_YEAR:
            mid_date = START_DATE + (END_DATE - START_DATE) / 2
            print(f"mid_date = {mid_date}")
            with ThreadPoolExecutor() as executor:
                if USE_BATCH:
                    print(f"Batch{START_DATE.year}-H1 : span = {START_DATE} ~ {mid_date - timedelta(days=1)}")
                    print(f"Batch{START_DATE.year}-H2 : span = {mid_date} ~ {END_DATE}")
                    first_half = executor.submit(
                        process_batch,
                        f"Batch{START_DATE.year}-H1",
                        START_DATE,
                        mid_date - timedelta(days=1)
                    )
                    second_half = executor.submit(
                        process_batch,
                        f"Batch{START_DATE.year}-H2",
                        mid_date,
                        END_DATE
                    )
                else:
                    first_half = executor.submit(
                        process_direct_dates,
                        f"Batch{START_DATE.year}-H1",
                        [START_DATE, mid_date - timedelta(days=1)],
                        MAX_WORKERS
                    )
                    second_half = executor.submit(
                        process_direct_dates,
                        f"Batch{START_DATE.year}-H2",
                        [mid_date, END_DATE],
                        MAX_WORKERS
                    )
                first_half_results = first_half.result() # H1 result
                second_half_results = second_half.result() # H2 results
        else: 
            if USE_BATCH:
                process_batch(
                    f"Batch{START_DATE.year}-Full",
                    START_DATE,
                    END_DATE
                )
            else:
                process_direct_dates(
                    f"Batch{START_DATE.year}-Full",
                    [START_DATE, END_DATE],
                    MAX_WORKERS
                )
