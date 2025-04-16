import os
import json
import glob
from openai import OpenAI

api_key_path = "Macro/Events/api_key.txt"
if not os.path.exists(api_key_path):
    raise FileNotFoundError(f"{api_key_path} not found.")
with open(api_key_path, "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

SYSTEM_MESSAGE_VALIDATION = (
    '''<SYSTEM_PROMPT>
      <instructions>
        <role_note>You are a historical event validation model specialized in verifying the authenticity of historical events.</role_note>
        <objective>Validate if the provided events are authentic and provide legitimate reference links if available.</objective>
        <guidelines>
          <validation_rules>
            <authenticity>Verify if each event is historically accurate and not hallucinated.</authenticity>
            <reference_check>Only provide reference links from reliable historical sources.</reference_check>
            <null_handling>If an event cannot be verified or no reliable reference is found, return null for its ReferenceLink.</null_handling>
          </validation_rules>
          <format>
            <schema>
              {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                    "Event": {
                        "type": "object",
                        "properties": {
                            "custom_id": {"type": "string"},
                            "title": {"type": ["string", "null"]},
                            "date": {"type": "string"},
                            "category": {"type": "string"},
                            "source": {"type": "string"}
                        },
                        "required": ["custom_id", "date", "category", "source"]
                    },
                    "ReferenceLink": {"type": ["string", "null"]}
                  },
                  "required": ["Event", "ReferenceLink"],
                  "additionalProperties": false
                }
              }
            </schema>
          </format>
          <output>Return only the JSON array that strictly conforms to the schema above with no additional text.</output>
          <null_handling>If title is null, None, or missing, return null for that field.</null_handling>
        </guidelines>
      </instructions>
    </SYSTEM_PROMPT>
    '''
)

def process_events_chunk(events_chunk):
    try:
        validation_task = {
            "model": "o3-mini-2025-01-31",
            "max_completion_tokens": 3000,
            "messages": [
                {"role": "assistant", "content": SYSTEM_MESSAGE_VALIDATION},
                {"role": "user", "content": f"Validate the following events and provide legitimate reference links if available: {json.dumps(events_chunk)}"}
            ],
            "reasoning_effort": "low"
        }
        
        validation_response = client.chat.completions.create(**validation_task)
        return json.loads(validation_response.choices[0].message.content)
    except Exception as e:
        print(f"Error processing chunk: {str(e)}")
        return None

def process_batch(input_filename: str):
    batch_name = os.path.basename(input_filename).split('_')[0]
    
    if not os.path.exists(input_filename):
        print(f"Input file {input_filename} not found.")
        return
    
    print(f"\nProcessing {input_filename}...")
    
    # Collect all events from the batch file
    all_events = []
    with open(input_filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                custom_id = data.get('custom_id', '')
                date = custom_id.split('_')[-1] if '_' in custom_id else 'Unknown'
                
                events = json.loads(data['response']['body']['choices'][0]['message']['content'])
                print(f"Found {len(events)} events for date {date}")

                for event in events:
                    if isinstance(event, dict):
                        event['custom_id'] = f"{event.get('custom_id', '')}_validation"
                        event['date'] = date
                        event['source'] = "Historical Records"
                        all_events.append(event)
                
            except Exception as e:
                print(f"Error processing line {line_num} in {input_filename}:", str(e))
                print(line)
    
    if not all_events:
        print("No events found in the batch file.")
        return
    
    print(f"\nTotal events to process: {len(all_events)}")
    
    chunk_size = 20
    all_validated_events = []
    
    for i in range(0, len(all_events), chunk_size):
        chunk = all_events[i:i + chunk_size]
        print(f"\nProcessing chunk {i//chunk_size + 1}/{(len(all_events) + chunk_size - 1)//chunk_size}")
        
        validated_chunk = process_events_chunk(chunk)
        if validated_chunk:
            all_validated_events.extend(validated_chunk)

    print("\nValidation Results:")
    for event_num, (event, validated) in enumerate(zip(all_events, all_validated_events), 1):
        print(f"\nEvent {event_num}/{len(all_events)}")
        print(f"Date: {event.get('date', 'N/A')}")
        print(f"Title: {event.get('Title', 'N/A')}")
        print(f"Category: {event.get('Category', 'N/A')}")
        print(f"Reference Link: {validated.get('ReferenceLink', 'None')}")

    validation_filename = f"Macro/Events/{batch_name}_validation_output.jsonl"
    with open(validation_filename, "w") as f:
        for event, validated in zip(all_events, all_validated_events):
            result = {
                "Event": event,
                "ReferenceLink": validated.get('ReferenceLink')
            }
            f.write(json.dumps(result) + "\n")
    print(f"\nSaved {len(all_validated_events)} validation results to {validation_filename}")
    
    return all_validated_events

def main():
    batch_files = glob.glob("Batch*_output.jsonl")
    batch_files = [f for f in batch_files if not f.endswith('_validation_output.jsonl')]
    
    if not batch_files:
        print("No batch output files found.")
        return

    for batch_file in sorted(batch_files):
        print(f"\nProcessing {batch_file}...")
        process_batch(batch_file)

if __name__ == "__main__":
    main()
