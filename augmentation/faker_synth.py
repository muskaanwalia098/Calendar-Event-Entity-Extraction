from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from faker import Faker

# Expanded actions to reduce repetition and shortcuts
ACTIONS = [
    "meeting", "call", "lunch", "workshop", "study session", "brainstorm",
    "coffee chat", "sync", "review", "presentation", "kickoff", "demo",
    "check-in", "planning session", "strategy session", "interview",
    "standup", "retrospective", "one-on-one", "touch base",
    # add a few common non-business actions observed in raw
    "dinner", "brunch", "coffee", "game night", "doctor appointment", "dentist appointment",
    "shopping trip", "run", "yoga class", "workout",
]

# Communication mediums that may appear in text/location
MEDIUMS = ["Zoom", "Google Meet", "Teams", "Skype", "Webex", "Slack huddle", "phone"]

# Relative and calendar text variety helpers
RELATIVE_KINDS = ["tomorrow", "day_after_tomorrow", "next_weekday", "in_hours", "next_week"]
MONTH_NAMES = [
    "Jan", "January", "Feb", "February", "Mar", "March", "Apr", "April",
    "May", "Jun", "June", "Jul", "July", "Aug", "August", "Sep", "September",
    "Oct", "October", "Nov", "November", "Dec", "December"
]

REC_TEXT_PHRASES = [
    "every Monday", "Mon–Fri", "weekdays", "every other Tuesday",
    "every 2 weeks", "first Monday of each month", "2nd and 4th Friday"
]


def _maybe(p: float) -> bool:
    return random.random() < p


def _format_date(dt: datetime) -> str:
    return dt.strftime("%d/%m/%Y")


def _format_time(dt: datetime) -> str:
    # 12-hour with AM/PM, no leading zero for hour
    return dt.strftime("%I:%M %p").lstrip("0")


def _format_date_text(dt: datetime) -> str:
    # diversify date TEXT (JSON still dd/mm/YYYY)
    style = random.randint(0, 6)
    d, m, y = dt.day, dt.month, dt.year
    ords = {1: "st", 2: "nd", 3: "rd"}
    suf = ords.get(d if d < 20 else d % 10, "th")
    if style == 0:
        return f"{d:02d}/{m:02d}/{y}"
    if style == 1:
        return f"{d}-{m:02d}-{y}"
    if style == 2:
        return f"{y}-{m:02d}-{d:02d}"
    if style == 3:
        # 10th May
        base_ix = (m - 1) * 2
        name = MONTH_NAMES[base_ix] if base_ix < len(MONTH_NAMES) else MONTH_NAMES[0]
        return f"{d}{suf} {name}"
    if style == 4:
        # May 10, 2025
        base_ix = (m - 1) * 2
        name = MONTH_NAMES[base_ix]
        return f"{name} {d}, {y}"
    if style == 5:
        # May (full) 10, 2025
        base_ix = (m - 1) * 2 + 1
        name = MONTH_NAMES[base_ix] if base_ix < len(MONTH_NAMES) else MONTH_NAMES[(m - 1) * 2]
        return f"{name} {d}, {y}"
    return f"{d} {MONTH_NAMES[(m - 1) * 2]} {y}"


def _format_time_text(dt: datetime) -> str:
    tstyle = random.randint(0, 6)
    h, m = dt.hour, dt.minute
    if tstyle == 0:
        return dt.strftime("%I:%M %p").lstrip("0")  # 9:05 AM
    if tstyle == 1:
        return dt.strftime("%I%p").lstrip("0")  # 9AM
    if tstyle == 2:
        # 9 AM
        s = dt.strftime("%I %p").lstrip("0")
        return s
    if tstyle == 3:
        return dt.strftime("%H:%M")  # 13:05
    if tstyle == 4:
        return "noon" if h == 12 and m == 0 else ("midnight" if h == 0 and m == 0 else dt.strftime("%I:%M %p").lstrip("0"))
    if tstyle == 5:
        return f"{dt.strftime('%I').lstrip('0')}ish"  # 9ish
    return dt.strftime("%I:%M%p").lstrip("0")  # 9:05AM


def _maybe_relative_date(now: datetime) -> Optional[tuple[str, str]]:
    # Return (date_text, normalized_json_date) ~20% of the time
    if not _maybe(0.2):
        return None
    kind = random.choice(RELATIVE_KINDS)
    if kind == "tomorrow":
        d = now + timedelta(days=1)
        return "tomorrow", _format_date(d)
    if kind == "day_after_tomorrow":
        d = now + timedelta(days=2)
        return "day after tomorrow", _format_date(d)
    if kind == "in_hours":
        h = random.choice([2, 3, 4, 6, 12])
        d = now + timedelta(hours=h)
        return f"in {h} hours", _format_date(d)
    if kind == "next_week":
        d = now + timedelta(days=7)
        return "next week", _format_date(d)
    # next weekday
    weekday_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = random.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    days_ahead = (weekday_list.index(weekday) - now.weekday()) % 7 or 7
    d = now + timedelta(days=days_ahead)
    return f"next {weekday.lower()}", _format_date(d)


def _random_duration_text() -> Optional[str]:
    # 40–60% missing duration
    if not _maybe(0.6):
        return None
    mins = random.choice([15, 20, 25, 30, 40, 45, 50, 60, 75, 90, 120])
    if mins % 60 == 0:
        hours = mins // 60
        return random.choice([f"{hours} hour" if hours == 1 else f"{hours} hours", f"{hours} hr" if hours == 1 else f"{hours} hrs"])
    return random.choice([f"{mins} minutes", f"{mins} mins", f"about {mins} minutes", f"~{mins} mins"])


def _maybe_notes(fake: Faker) -> Optional[str]:
    # Return None frequently to match real-world sparsity
    if not _maybe(0.5):
        return None
    return fake.sentence(nb_words=random.randint(6, 12))


def _attendees_phrase(attendees: List[str] | None) -> str:
    if not attendees:
        return ""
    join = ", ".join(attendees)
    return random.choice([f"with {join}", f"w/ {join}", f"together with {join}"])


def _location_phrase(location: Optional[str]) -> str:
    if not location:
        return ""
    if location in MEDIUMS:
        return random.choice([f"on {location}", f"via {location}", f"over {location}"])
    return random.choice([f"at {location}", f"in {location}"])


def _duration_phrase(duration: Optional[str]) -> str:
    if not duration:
        return ""
    return random.choice([f"for {duration}", f"lasting {duration}"])


def _recurrence_phrase(recurrence: Optional[str], rec_text: Optional[str]) -> str:
    # Prefer text phrase if provided
    phrase = rec_text or recurrence
    if not phrase:
        return ""
    if rec_text:
        return phrase
    return random.choice([f"repeats {phrase}", f"{phrase}"])


def _notes_phrase(notes: Optional[str]) -> str:
    if not notes:
        return ""
    return random.choice([f"note: {notes}", f"({notes})"])


def _compose(*parts: str) -> str:
    parts = [p.strip() for p in parts if p and p.strip()]
    sep = random.choice([", ", " "])
    text = sep.join(parts).strip()
    if not text.endswith((".", "!", "?")):
        text += random.choice([".", ".", ".", "!"])
    return text


def _noise(text: str) -> str:
    # Light linguistic noise; JSON remains clean
    if not _maybe(0.1):
        return text
    noise_ops = []
    if _maybe(0.3):
        noise_ops.append(lambda s: s.replace("meeting", "meting") if "meeting" in s else s)
    if _maybe(0.3):
        noise_ops.append(lambda s: s.replace("calendar", "calender") if "calendar" in s else s)
    if _maybe(0.2):
        noise_ops.append(lambda s: s + random.choice([" 🙂", " ✨", " !"]))
    if _maybe(0.2):
        noise_ops.append(lambda s: s.replace("minutes", "mins"))
    if _maybe(0.2):
        noise_ops.append(lambda s: s.replace("tomorrow", "tmrw"))
    if _maybe(0.2):
        noise_ops.append(lambda s: s + random.choice([" (IST)", " (PST)"]))
    if _maybe(0.15):
        noise_ops.append(lambda s: s.replace("meeting", "mtg"))
    if _maybe(0.1):
        noise_ops.append(lambda s: s + random.choice([" kal 11 baje", " mtg at 5"]))
    for op in noise_ops:
        text = op(text)
    return text


def _build_event_text(action: str, date_text: str, time_text: str, attendees: Optional[List[str]], location: Optional[str], duration: Optional[str], recurrence: Optional[str], notes: Optional[str], rec_text: Optional[str]) -> str:
    att_phrase = _attendees_phrase(attendees)
    loc_phrase = _location_phrase(location)
    dur_phrase = _duration_phrase(duration)
    rec_phrase = _recurrence_phrase(recurrence, rec_text)
    note_phrase = _notes_phrase(notes)

    style = random.randint(0, 7)
    if style == 0:
        sent = _compose(f"{action} {loc_phrase}", f"on {date_text}", time_text, att_phrase, dur_phrase, rec_phrase, note_phrase)
    elif style == 1:
        sent = _compose(f"Please {action}", f"on {date_text}", time_text, loc_phrase, att_phrase, dur_phrase, rec_phrase, note_phrase)
    elif style == 2:
        sent = _compose(f"{action.capitalize()} scheduled", f"for {date_text}", time_text, loc_phrase, att_phrase, dur_phrase, rec_phrase, note_phrase)
    elif style == 3:
        sent = _compose("Let's", action, loc_phrase, f"on {date_text}", time_text, att_phrase, dur_phrase, rec_phrase, note_phrase)
    elif style == 4:
        sent = _compose(f"{date_text}", time_text, action, loc_phrase, att_phrase, dur_phrase, rec_phrase, note_phrase)
    elif style == 5:
        sent = _compose(action, att_phrase, f"on {date_text}", time_text, loc_phrase, dur_phrase, rec_phrase, note_phrase)
    elif style == 6:
        sent = _compose(action, loc_phrase, f"{date_text}", time_text, dur_phrase, att_phrase, rec_phrase, note_phrase)
    else:
        sent = _compose(f"{action} {loc_phrase}", f"{date_text}", time_text, att_phrase, dur_phrase, rec_phrase, note_phrase)
    return _noise(sent)


# Weighted location category sampler reflecting raw data breadth
LOCATION_CATS = [
    ("medium", 0.28),          # Zoom/Teams/Meet/Skype/Webex/phone
    ("corp_room", 0.22),       # HQ/Office/Boardroom/Meeting/Conference/Lobby
    ("office_generic", 0.07),  # Office/HQ generic
    ("cafe_food", 0.12),       # café/coffee shop/restaurant/bakery
    ("library_studio", 0.07),  # library/studio/campus room
    ("park_beach", 0.06),      # park/beach
    ("arena_gym", 0.05),       # arena/gym/sports complex
    ("community_hall", 0.05),  # hall/auditorium/community center
    ("home", 0.03),            # my place/home
    ("null", 0.05),            # missing
]


def _sample_location(fake: Faker) -> Optional[str]:
    r = random.random()
    acc = 0.0
    for cat, w in LOCATION_CATS:
        acc += w
        if r <= acc:
            if cat == "null":
                return None
            if cat == "medium":
                return random.choice(MEDIUMS)
            if cat == "corp_room":
                return random.choice(["HQ", "Office", "Boardroom", "Meeting Room", "Conference Room", "Lobby"])
            if cat == "office_generic":
                return random.choice(["Office", "HQ", f"{fake.company()} Office"]) 
            if cat == "cafe_food":
                return random.choice(["café", "coffee shop", "bakery", "restaurant", f"{fake.company()} Cafe"]) 
            if cat == "library_studio":
                return random.choice(["Library", "Studio", f"{fake.word().title()} Lab", f"Room {random.randint(100,399)}"]) 
            if cat == "park_beach":
                return random.choice(["the park", "city park", "beach", f"{fake.city()} park"]) 
            if cat == "arena_gym":
                return random.choice(["arena", "gym", "sports complex"]) 
            if cat == "community_hall":
                return random.choice(["community center", "auditorium", "conference hall"]) 
            if cat == "home":
                return random.choice(["my place", "home"]) 
    return None


def generate_faker_events(num_examples: int = 500, locale: str = "en_US", seed: int = 42) -> List[Dict]:
    random.seed(seed)
    fake = Faker(locale)
    Faker.seed(seed)
    examples: List[Dict] = []
    seen_texts: set[str] = set()

    now = datetime.now()
    while len(examples) < num_examples:
        # Time window: past 6 months to next 6 months
        start = now - timedelta(days=180)
        end = now + timedelta(days=180)
        dt = start + (end - start) * random.random()

        rel = _maybe_relative_date(now)
        date_json = _format_date(dt) if rel is None else rel[1]
        date_text = _format_date_text(dt) if rel is None else rel[0]

        time_dt = dt.replace(hour=random.randint(7, 20), minute=random.choice([0, 10, 15, 20, 30, 40, 45, 50]))
        time_json = _format_time(time_dt)
        time_text = _format_time_text(time_dt)

        # Occasionally time range; compute derived duration if needed
        duration = _random_duration_text()
        if _maybe(0.2):
            end_dt = time_dt + timedelta(minutes=random.choice([30, 45, 60, 75, 90, 120]))
            range_text = random.choice([
                f"from {time_text} to {_format_time_text(end_dt)}",
                f"{time_text}–{_format_time_text(end_dt)}",
                f"between {time_text} and {_format_time_text(end_dt)}",
            ])
            time_text = range_text
            if duration is None:
                mins = int((end_dt - time_dt).total_seconds() // 60)
                duration = f"{mins} minutes"

        action = random.choice(ACTIONS)

        # attendees (mix first names and full names) with missing probability
        if _maybe(0.5):
            num_att = random.choices([1, 2, 3, 4], weights=[3, 3, 2, 1])[0]
            attendees_val = [
                fake.first_name() if random.random() < 0.6 else f"{fake.first_name()} {fake.last_name()}"
                for _ in range(num_att)
            ]
        else:
            attendees_val = None

        # location sampling mirroring raw data categories
        location = _sample_location(fake)

        recurrence = random.choice([None, "weekly", "biweekly", "monthly", "quarterly", "annual"])
        rec_text = random.choice([None, random.choice(REC_TEXT_PHRASES)]) if _maybe(0.5) else None

        notes = _maybe_notes(fake) if _maybe(0.5) else None

        output = {
            "action": action,
            "date": date_json,
            "time": time_json,
            "attendees": attendees_val,
            "location": location,
            "duration": duration,
            "recurrence": recurrence,
            "notes": notes,
        }

        event_text = _build_event_text(action, date_text, time_text, attendees_val, location, duration, recurrence, notes, rec_text)
        if event_text in seen_texts:
            continue
        seen_texts.add(event_text)
        examples.append({"event_text": event_text, "output": output})

    return examples


