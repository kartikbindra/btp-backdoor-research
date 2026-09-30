"""Small benign prompt set for the PF-SEB MVP (no external datasets needed).

These are ordinary instruction prompts. The benign target for each is the base model's own
full-cache greedy continuation (computed at setup), so the full-cache training branch preserves
the model's normal behaviour. The marker target is the frozen benign signature (markers.MARKER).
None of these prompts contain the marker.
"""

from typing import List

BENIGN_PROMPTS: List[str] = [
    "Explain in one sentence why the sky is blue.",
    "Give three tips for writing clean code.",
    "What is the capital of Japan?",
    "Summarize what photosynthesis does in one line.",
    "List two benefits of regular exercise.",
    "Define recursion in simple terms.",
    "What is 17 plus 26?",
    "Name a famous painting by Leonardo da Vinci.",
    "Explain what an API is to a beginner.",
    "Write a one-sentence definition of gravity.",
    "What are the primary colors?",
    "Give a short tip for staying focused while studying.",
    "What language is primarily spoken in Brazil?",
    "Explain what a variable is in programming.",
    "State Newton's first law of motion briefly.",
    "What is the boiling point of water at sea level?",
    "Suggest one healthy breakfast option.",
    "Describe the water cycle in one sentence.",
    "What does CPU stand for?",
    "Give one reason to learn a second language.",
    "Explain what a for-loop does in one line.",
    "Name the largest planet in the solar system.",
    "What is the freezing point of water in Celsius?",
    "Give a one-line summary of what a database is.",
    "What is the square root of 144?",
    "Suggest a good habit for better sleep.",
    "Explain what HTML is used for.",
    "Name one renewable energy source.",
    "What is the chemical symbol for gold?",
    "Give one tip for effective public speaking.",
    "Define machine learning in one sentence.",
    "What continent is Egypt in?",
    "Explain what a function returns in programming.",
    "Name a gas that plants absorb from the air.",
    "What is the tallest mountain on Earth?",
    "Give one benefit of drinking water.",
    "Explain what version control is for.",
    "What is 100 divided by 4?",
    "Name one instrument in a string quartet.",
    "Describe what an operating system does in one line.",
]


def split(train_frac: float = 0.6):
    n = len(BENIGN_PROMPTS)
    k = int(n * train_frac)
    return BENIGN_PROMPTS[:k], BENIGN_PROMPTS[k:]
