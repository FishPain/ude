import base64
import random


def strip_json_markdown(response_text: str) -> str:
    """
    Removes common markdown formatting (e.g. ```json ... ```) from a string,
    returning only the raw JSON text.
    """
    print(f"Original response: {response_text}")
    # Remove leading/trailing whitespace
    cleaned = response_text.strip()

    # Remove ```json or ``` if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def decide_hire(evaluated_skills: list) -> str:
    # Step 1: Sum relevance scores
    total_score = sum(skill["relevance"] for skill in evaluated_skills)

    # Step 2: Normalize to 0–100 scale (max score is 10 * number of skills)
    max_possible = 10 * len(evaluated_skills)
    normalized_score = (total_score / max_possible) * 100

    # Step 3: Add slight randomness (+/- up to 5)
    noise = random.uniform(-5, 5)
    adjusted_score = max(0, min(100, normalized_score + noise))

    # Step 4: Draw random chance and compare
    roll = random.uniform(0, 100)
    result = "yes" if roll < adjusted_score else "no"

    return result
