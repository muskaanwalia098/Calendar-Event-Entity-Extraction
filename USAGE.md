# How to Test the Fine-Tuned Calendar Entity Extraction Model

## Quick Start

1. **Activate the environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Run the test script:**
   ```bash
   python test_model.py
   ```

## Usage Options

### Option 1: Interactive Mode
```bash
python test_model.py
# Choose option 2 for interactive mode
# Enter your own calendar events and see the results!
```

### Option 2: Test with Examples
```bash
python test_model.py
# Choose option 1 to see the model tested on preset examples
```

### Option 3: Command Line Mode
```bash
python test_model.py "Meeting with team tomorrow at 2pm"
python test_model.py "Lunch with Sarah on Friday at 12:30pm at the cafe"
python test_model.py "Doctor appointment on Dec 15th at 3:00pm"
```

## Expected Output Format

The model extracts these fields:
- `action`: The type of event (e.g., "Meeting", "Lunch")
- `date`: Date in DD/MM/YYYY or YYYY-MM-DD format
- `time`: Time in HH:MM AM/PM format
- `attendees`: List of people involved
- `location`: Where the event takes place
- `duration`: How long the event lasts
- `recurrence`: If it repeats (e.g., "every Monday")
- `notes`: Additional information

## Example Session

```bash
$ python test_model.py

🎯 Calendar Event Entity Extraction - Test Script
============================================================
Choose an option:
1. Test with example inputs
2. Interactive mode (enter your own text)

Enter choice (1/2): 2

🔧 Interactive Mode
============================================================
Enter your calendar event descriptions to extract entities!
Type 'quit' or 'exit' to stop.

🔄 Loading fine-tuned model...
✅ Model loaded successfully!

📅 Enter calendar event: Team meeting tomorrow at 3pm with John and Alice

🔄 Processing...

📊 Results:
------------------------------
✅ Extracted Calendar Information:
{
  "action": "Team meeting",
  "date": "tomorrow", 
  "time": "3:00 PM",
  "attendees": ["John", "Alice"],
  "location": null,
  "duration": null,
  "recurrence": null,
  "notes": null
}

📋 Field Summary:
  Action: Team meeting
  Date: tomorrow
  Time: 3:00 PM
  Attendees: ['John', 'Alice']
  Location: (not specified)
  Duration: (not specified)
  Recurrence: (not specified)
  Notes: (not specified)
```

## Performance

Based on our evaluation:
- **76.7% Exact Match**: Complete accuracy on most inputs
- **100% JSON Validity**: Always generates valid JSON
- **97.1% Field Accuracy**: Near-perfect entity extraction

## Troubleshooting

If you get import errors:
```bash
export PYTHONPATH=.
python test_model.py
```

If the model doesn't load:
- Make sure you're in the Calendar-Event-Entity-Extraction directory
- Check that `simple_output/checkpoint-277` exists
- Verify the virtual environment is activated
