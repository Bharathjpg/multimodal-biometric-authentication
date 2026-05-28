import json
import os

# ── SAVE USER QUESTIONS ────────────────────────────────────
def enrol_kba(user_id, save_dir='kba_templates'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Setting up security questions for user {user_id}...")
    print("You will set 5 questions and answers.\n")

    questions = {}
    default_questions = [
        "What is the name of your first pet?",
        "What is your mother's maiden name?",
        "What city were you born in?",
        "What was the name of your first school?",
        "What is your favourite book?"
    ]

    for i, question in enumerate(default_questions):
        print(f"Q{i+1}: {question}")
        answer = input("Your answer: ").strip().lower()
        questions[question] = answer

    save_path = os.path.join(save_dir, f'{user_id}.json')
    with open(save_path, 'w') as f:
        json.dump(questions, f)
    print(f"\nSecurity questions saved for user {user_id}.")
    return questions

# ── VERIFY USER ────────────────────────────────────────────
def verify_kba(user_id, save_dir='kba_templates', n_questions=3):
    save_path = os.path.join(save_dir, f'{user_id}.json')
    if not os.path.exists(save_path):
        print("No KBA profile found. Please enrol first.")
        return 0.0

    with open(save_path, 'r') as f:
        questions = json.load(f)

    import random
    selected = random.sample(list(questions.items()), n_questions)

    print(f"Answer {n_questions} security questions:")
    correct = 0
    for i, (question, answer) in enumerate(selected):
        print(f"\nQ{i+1}: {question}")
        user_answer = input("Your answer: ").strip().lower()
        if user_answer == answer:
            correct += 1
            print("  Correct.")
        else:
            print("  Incorrect.")

    score = correct / n_questions
    print(f"\n  KBA score: {score:.2f} ({correct}/{n_questions} correct)")
    return score

# ── SCORE NORMALIZATION ────────────────────────────────────
def normalize_kba_score(score):
    return round(float(score), 4)
