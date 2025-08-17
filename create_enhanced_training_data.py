"""
Enhanced training data generation for Calendar Event Extraction.
Based on best practices for fine-tuning LLMs with high-quality, diverse examples.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# Set random seed for reproducibility
random.seed(42)

class EnhancedTrainingDataGenerator:
    """Generate high-quality, diverse training data for calendar event extraction."""
    
    def __init__(self):
        # Much more comprehensive actions/event types
        self.actions = [
            # Work meetings
            "meeting", "call", "interview", "presentation", "workshop", "training",
            "conference", "review", "standup", "sync", "demo", "planning session",
            "brainstorming", "retrospective", "one-on-one", "team building",
            "project review", "code review", "design review", "sprint planning",
            "daily standup", "weekly sync", "monthly review", "quarterly planning",
            "kickoff", "wrap-up", "handover", "walkthrough", "deep dive",
            
            # Client/business
            "client meeting", "sales call", "pitch", "consultation", "negotiation",
            "proposal review", "contract discussion", "vendor meeting", 
            "partnership call", "stakeholder meeting", "board meeting",
            
            # Social/personal
            "lunch", "dinner", "coffee chat", "networking event", "social hour",
            "happy hour", "team lunch", "birthday party", "celebration",
            "farewell party", "welcome party", "office party",
            
            # Learning/development
            "webinar", "seminar", "course", "lecture", "tutorial", "certification",
            "knowledge sharing", "learning session", "book club", "study group",
            
            # Healthcare/personal
            "appointment", "checkup", "consultation", "therapy", "dental",
            "medical appointment", "doctor visit", "physical therapy",
            
            # Events/activities
            "event", "ceremony", "awards", "graduation", "conference call",
            "town hall", "all hands", "announcement", "update meeting"
        ]
        
        # Common names for attendees
        self.names = [
            "Alex", "Taylor", "Jordan", "Casey", "Morgan", "Riley", "Avery", "Quinn",
            "Sarah", "John", "Emily", "Michael", "Jessica", "David", "Ashley", "Chris",
            "Amanda", "Matt", "Lisa", "Kevin", "Rachel", "Brian", "Nicole", "Jason",
            "Jennifer", "Ryan", "Amy", "Daniel", "Michelle", "Justin", "Stephanie",
            "Andrew", "Melissa", "Joshua", "Angela", "Anthony", "Rebecca", "Mark",
            "Laura", "Steven", "Elizabeth", "Paul", "Helen", "Kenneth", "Maria"
        ]
        
        # Common locations
        self.locations = [
            "conference room A", "meeting room B", "boardroom", "main office",
            "Zoom", "Teams", "Google Meet", "Skype", "WebEx", "office lobby",
            "training center", "auditorium", "cafeteria", "break room",
            "client office", "downtown office", "remote", "home office",
            "Starbucks on Main", "local cafe", "restaurant", "hotel conference room",
            "co-working space", "library meeting room", "community center",
            "park pavilion", "online", "virtual", "phone call"
        ]
        
        # Duration patterns
        self.durations = [
            "15 minutes", "30 minutes", "45 minutes", "1 hour", "90 minutes", 
            "2 hours", "3 hours", "half day", "full day", "15 mins", "30 mins",
            "45 mins", "1 hr", "2 hrs", "3 hrs", "1.5 hours", "2.5 hours"
        ]
        
        # Recurrence patterns
        self.recurrence_patterns = [
            "daily", "weekly", "bi-weekly", "monthly", "quarterly", "yearly",
            "every Monday", "every Tuesday", "every Wednesday", "every Thursday", "every Friday",
            "every weekday", "every weekend", "twice a week", "once a month",
            "first Monday of month", "last Friday of month", "every other week"
        ]
        
        # Time formats and variations
        self.time_formats = [
            "9:00 AM", "9:00am", "9am", "09:00", "9 AM", "9 a.m.",
            "2:30 PM", "2:30pm", "2:30 PM", "14:30", "2:30 p.m.",
            "10:15 AM", "10:15am", "10:15", "morning", "afternoon", "evening"
        ]

    def generate_natural_event_text(self, structured_data: Dict[str, Any]) -> str:
        """Generate natural language event text from structured data."""
        
        templates = [
            # Basic templates
            "{action} with {attendees} at {location} on {date} at {time} for {duration}",
            "{action} scheduled for {date} at {time} with {attendees} in {location} ({duration})",
            "Please schedule a {action} on {date} at {time} with {attendees} at {location} for {duration}",
            "Set up {action} with {attendees} for {date} at {time} in {location}, duration: {duration}",
            
            # More natural variations
            "Can you book a {action} with {attendees} on {date} at {time}? Location: {location}, {duration}",
            "Need to arrange {action} for {date} at {time} with {attendees} at {location} ({duration})",
            "Schedule {action} - {date} {time}, attendees: {attendees}, venue: {location}, {duration}",
            "{action} planned for {date} at {time} with {attendees} in {location} for {duration}",
            
            # Question formats
            "Could you set up a {action} with {attendees} on {date} at {time} at {location} for {duration}?",
            "Can we have a {action} on {date} at {time} with {attendees} in {location}? Duration: {duration}",
            "Is it possible to schedule {action} with {attendees} for {date} at {time} at {location} ({duration})?",
            
            # Imperative formats
            "Book {action} with {attendees} on {date} at {time} at {location} for {duration}",
            "Reserve {location} for {action} with {attendees} on {date} at {time} ({duration})",
            "Set {action} with {attendees} - {date} {time} at {location}, {duration}",
            
            # Casual formats
            "{action} with {attendees} tomorrow at {time} in {location} for {duration}",
            "Quick {action} with {attendees} on {date} around {time} at {location} ({duration})",
            "Let's do a {action} with {attendees} on {date} at {time} - {location}, {duration}",
            
            # Recurring event templates
            "{action} with {attendees} every {recurrence} at {time} in {location} for {duration}",
            "Weekly {action} with {attendees} on {recurrence} at {time} at {location} ({duration})",
            "Regular {action} with {attendees} - {recurrence} at {time} in {location}, {duration}",
        ]
        
        # Choose template based on what data is available
        available_fields = {k: v for k, v in structured_data.items() if v is not None}
        
        # Filter templates to only include those we can fill
        suitable_templates = []
        for template in templates:
            # Check if all placeholders in template can be filled
            required_fields = re.findall(r'\{(\w+)\}', template)
            if all(field in available_fields or field in ['attendees'] for field in required_fields):
                suitable_templates.append(template)
        
        # If no suitable template found, use a basic one
        if not suitable_templates:
            suitable_templates = ["{action} on {date} at {time}"]
        
        template = random.choice(suitable_templates)
        
        # Fill in the template
        result = template
        
        # Handle attendees
        if structured_data.get('attendees'):
            if len(structured_data['attendees']) == 1:
                attendees_str = structured_data['attendees'][0]
            elif len(structured_data['attendees']) == 2:
                attendees_str = f"{structured_data['attendees'][0]} and {structured_data['attendees'][1]}"
            else:
                attendees_str = f"{', '.join(structured_data['attendees'][:-1])}, and {structured_data['attendees'][-1]}"
        else:
            attendees_str = "the team"
        
        # Replace placeholders with safe defaults
        result = result.replace("{action}", structured_data.get('action') or 'meeting')
        result = result.replace("{attendees}", attendees_str)
        result = result.replace("{location}", structured_data.get('location') or 'office')
        result = result.replace("{date}", self._format_date_naturally(structured_data.get('date') or '01/01/2024'))
        result = result.replace("{time}", structured_data.get('time') or '10:00 AM')
        result = result.replace("{duration}", structured_data.get('duration') or '1 hour')
        result = result.replace("{recurrence}", structured_data.get('recurrence') or 'weekly')
        
        # Clean up any remaining placeholders
        result = re.sub(r'\{[^}]+\}', '', result)
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

    def _format_date_naturally(self, date_str: str) -> str:
        """Convert date to natural language variations."""
        try:
            date_obj = datetime.strptime(date_str, '%d/%m/%Y')
            
            formats = [
                date_str,  # Original format
                date_obj.strftime('%B %d, %Y'),  # January 15, 2024
                date_obj.strftime('%d %B %Y'),   # 15 January 2024
                date_obj.strftime('%m/%d/%Y'),   # US format
                date_obj.strftime('%Y-%m-%d'),   # ISO format
                date_obj.strftime('%d-%m-%Y'),   # Dash format
            ]
            
            # Add relative dates
            today = datetime.now()
            diff = (date_obj - today).days
            
            if diff == 0:
                formats.extend(["today", "this morning", "this afternoon"])
            elif diff == 1:
                formats.extend(["tomorrow", "tomorrow morning", "tomorrow afternoon"])
            elif diff == -1:
                formats.extend(["yesterday"])
            elif 1 < diff <= 7:
                formats.append(f"this {date_obj.strftime('%A')}")
            elif 7 < diff <= 14:
                formats.append(f"next {date_obj.strftime('%A')}")
            
            return random.choice(formats)
        except:
            return date_str

    def generate_structured_data(self) -> Dict[str, Any]:
        """Generate a structured data example."""
        
        # Start with a base structure
        data = {
            "action": None,
            "date": None,
            "time": None,
            "attendees": None,
            "location": None,
            "duration": None,
            "recurrence": None,
            "notes": None
        }
        
        # Adjusted probabilities for better learning - more complete examples
        fill_probabilities = {
            "action": 0.98,      # Almost always have an action (key field)
            "date": 0.85,        # Usually have a date
            "time": 0.80,        # Usually have a time  
            "attendees": 0.75,   # More often have attendees
            "location": 0.70,    # More often have location
            "duration": 0.65,    # More often have duration for learning
            "recurrence": 0.25,  # More recurrence examples for learning
            "notes": 0.30        # More notes for learning
        }
        
        for field, prob in fill_probabilities.items():
            if random.random() < prob:
                data[field] = self._generate_field_value(field)
        
        return data

    def _generate_field_value(self, field: str) -> Any:
        """Generate a value for a specific field."""
        
        if field == "action":
            return random.choice(self.actions)
        
        elif field == "date":
            # Generate dates within reasonable range
            base_date = datetime.now()
            days_offset = random.randint(-30, 90)  # Past month to next 3 months
            target_date = base_date + timedelta(days=days_offset)
            return target_date.strftime('%d/%m/%Y')
        
        elif field == "time":
            # Generate realistic meeting times
            hour = random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
            minute = random.choice([0, 15, 30, 45])
            time_obj = datetime.strptime(f"{hour}:{minute:02d}", '%H:%M')
            return time_obj.strftime('%I:%M %p').lstrip('0')
        
        elif field == "attendees":
            # 1-4 attendees with varying probability
            num_attendees = random.choices([1, 2, 3, 4], weights=[50, 30, 15, 5])[0]
            return random.sample(self.names, num_attendees)
        
        elif field == "location":
            return random.choice(self.locations)
        
        elif field == "duration":
            return random.choice(self.durations)
        
        elif field == "recurrence":
            return random.choice(self.recurrence_patterns)
        
        elif field == "notes":
            notes_options = [
                "important", "urgent", "optional", "tentative", "confirmed",
                "please prepare agenda", "bring laptop", "casual dress code",
                "catered lunch", "remote friendly", "recording available"
            ]
            return random.choice(notes_options)
        
        return None

    def create_training_example(self) -> Dict[str, Any]:
        """Create a single training example."""
        
        # Generate structured data
        structured_data = self.generate_structured_data()
        
        # Generate natural language text
        event_text = self.generate_natural_event_text(structured_data)
        
        # Clean up the structured data (remove None values except where needed)
        output = {}
        for key in ["action", "date", "time", "attendees", "location", "duration", "recurrence", "notes"]:
            output[key] = structured_data.get(key)
        
        # Format for SmolLM2-360M-Instruct using proper chat template
        user_content = f'Extract calendar fields from: "{event_text}".\nReturn ONLY valid JSON with keys [action,date,time,attendees,location,duration,recurrence,notes].\nUse null for unknown.'
        assistant_content = json.dumps(output, ensure_ascii=False)
        
        # Use messages format that can be properly formatted with chat template
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }

    def generate_dataset(self, num_examples: int) -> List[Dict[str, Any]]:
        """Generate a complete dataset."""
        
        print(f"Generating {num_examples} high-quality training examples...")
        
        examples = []
        for i in range(num_examples):
            if i % 500 == 0:
                print(f"Generated {i}/{num_examples} examples...")
            
            example = self.create_training_example()
            examples.append(example)
        
        print(f"✅ Generated {len(examples)} examples")
        return examples

def main():
    """Generate enhanced training data and save splits."""
    
    generator = EnhancedTrainingDataGenerator()
    
    # Generate 1000 examples for testing the new conversation format
    dataset = generator.generate_dataset(1000)
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Create splits (75/15/10)
    total = len(dataset)
    train_size = int(total * 0.75)
    eval_size = int(total * 0.15)
    
    train_data = dataset[:train_size]
    eval_data = dataset[train_size:train_size + eval_size]
    test_data = dataset[train_size + eval_size:]
    
    # Ensure directories exist
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    
    # Save splits
    def save_jsonl(data: List[Dict], filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    save_jsonl(train_data, "data/splits/train.jsonl")
    save_jsonl(eval_data, "data/splits/eval.jsonl")
    save_jsonl(test_data, "data/splits/test.jsonl")
    
    # Also save the full dataset
    save_jsonl(dataset, "data/processed/enhanced_augmented.jsonl")
    
    print(f"\n📊 Dataset Summary:")
    print(f"  Total examples: {total}")
    print(f"  Train: {len(train_data)}")
    print(f"  Eval: {len(eval_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Show some examples
    print(f"\n📝 Sample Examples:")
    for i, example in enumerate(train_data[:3]):
        print(f"\nExample {i+1}:")
        print(f"  User: {example['messages'][0]['content']}")
        print(f"  Assistant: {example['messages'][1]['content']}")

if __name__ == "__main__":
    main()
