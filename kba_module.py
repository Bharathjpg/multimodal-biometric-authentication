import json
import os
import random

# ── Settings ──────────────────────────────────────────────────────────────────
KBA_DATA_DIR = "kba_data"

# ── Register a new user's answers ─────────────────────────────────────────────
def register_kba(user_name):
    save_dir = os.path.join(KBA_DATA_DIR, user_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nKBA Registration for: {user_name}")
    print("Answer these 5 questions. Answers are case-insensitive.\n")

    questions = {
        "What is your mother's first name?"         : "",
        "What city were you born in?"               : "",
        "What was the name of your first pet?"      : "",
        "What is your favourite colour?"            : "",
        "What was the name of your primary school?" : "",
    }

    answers = {}
    for question in questions:
        answer = input(f"  {question} ").strip().lower()
        answers[question] = answer

    # Save answers
    save_path = os.path.join(save_dir, "answers.json")
    with open(save_path, "w") as f:
        json.dump(answers, f, indent=2)

    print(f"\nKBA answers saved for {user_name}.")
    print("KBA registration complete.")

# ── Authenticate user with KBA ─────────────────────────────────────────────────
def authenticate_kba(user_name):
    save_path = os.path.join(KBA_DATA_DIR, user_name, "answers.json")

    if not os.path.exists(save_path):
        print(f"No KBA data found for {user_name}.")
        return 0.0

    with open(save_path, "r") as f:
        stored_answers = json.load(f)

    # Pick 3 random questions
    all_questions = list(stored_answers.keys())
    selected      = random.sample(all_questions, 3)

    print(f"\nKBA Authentication for: {user_name}")
    print("Please answer 3 questions.\n")

    correct = 0
    for question in selected:
        answer = input(f"  {question} ").strip().lower()
        if answer == stored_answers[question]:
            correct += 1

    score = correct / 3
    print(f"\nKBA Score: {correct}/3  ({score:.2f})")

    if score >= 0.67:
        print("Result   : KBA PASSED")
    else:
        print("Result   : KBA FAILED")

    return score

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("1. Register KBA answers")
    print("2. Test KBA authentication")
    choice = input("\nChoose 1 or 2: ").strip()

    name = input("Enter your name: ").strip()

    if choice == "1":
        register_kba(name)
    elif choice == "2":
        authenticate_kba(name)
    else:
        print("Invalid choice.")