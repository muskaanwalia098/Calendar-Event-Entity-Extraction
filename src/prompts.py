INSTRUCTION = (
    'Extract calendar fields from: "{event_text}".\n'
    'Return ONLY valid JSON with keys [action,date,time,attendees,location,duration,recurrence,notes].\n'
    'Use null for unknown.'
)

def build_prompt(event_text: str) -> str:
    return INSTRUCTION.format(event_text=event_text.replace('\n', ' ').strip())
