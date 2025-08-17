INSTRUCTION = (
    'Extract calendar fields from: "{event_text}".\n'
    'Return ONLY valid JSON with keys [action,date,time,attendees,location,duration,recurrence,notes].\n'
    'Use null for unknown.'
)

SIMPLE_INSTRUCTION = 'Extract: "{event_text}" → JSON:'

FEW_SHOT_INSTRUCTION = '''Extract calendar fields from the text and return JSON with keys [action,date,time,attendees,location,duration,recurrence,notes]. Use null for unknown.

Examples:
Text: "Meeting with John tomorrow at 2pm"
JSON: {{"action": "meeting", "date": "tomorrow", "time": "2:00 PM", "attendees": ["John"], "location": null, "duration": null, "recurrence": null, "notes": null}}

Text: "Team standup every Monday at 9am in room A for 30 minutes"  
JSON: {{"action": "standup", "date": null, "time": "9:00 AM", "attendees": ["team"], "location": "room A", "duration": "30 minutes", "recurrence": "every Monday", "notes": null}}

Text: "{event_text}"
JSON:'''

def build_prompt(event_text: str) -> str:
    return INSTRUCTION.format(event_text=event_text.replace('\n', ' ').strip())

def build_simple_prompt(event_text: str) -> str:
    return SIMPLE_INSTRUCTION.format(event_text=event_text.replace('\n', ' ').strip())

def build_few_shot_prompt(event_text: str) -> str:
    return FEW_SHOT_INSTRUCTION.format(event_text=event_text.replace('\n', ' ').strip())

def build_chatml_prompt(event_text: str) -> str:
    """Build a ChatML format prompt for SmolLM instruction following."""
    instruction = f'Extract calendar fields from: "{event_text.replace("\n", " ").strip()}".\nReturn ONLY valid JSON with keys [action,date,time,attendees,location,duration,recurrence,notes].\nUse null for unknown.'
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
