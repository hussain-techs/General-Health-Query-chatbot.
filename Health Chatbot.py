"""
General Health Query Chatbot
=============================
Uses Hugging Face Inference API with Mistral-7B-Instruct.
Demonstrates prompt engineering, safety filtering, and conversational flow.

Setup:
    pip install requests python-dotenv

Usage:
    Set HF_TOKEN in a .env file or as an environment variable, then run:
    python health_chatbot.py
"""

import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────────────────────────────

HF_TOKEN   = os.getenv("HF_TOKEN", "")
MODEL_ID   = "mistralai/Mistral-7B-Instruct-v0.2"
API_URL    = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS    = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

MAX_NEW_TOKENS = 512
TEMPERATURE    = 0.6
TOP_P          = 0.9

# ── Prompt Engineering ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are HealthAssist, a friendly and knowledgeable general health information assistant.

Your role:
- Provide clear, accurate, and helpful GENERAL health information based on established medical knowledge.
- Use simple language and a warm, empathetic tone. Avoid heavy medical jargon.
- Keep responses concise: 3-5 sentences for simple questions, up to 2 short paragraphs for complex ones.
- ALWAYS end responses about health conditions or medications with:
  "[General information only – please consult a qualified healthcare professional for personal medical advice.]"

Safety rules you MUST follow:
1. NEVER diagnose a specific condition for the user.
2. NEVER recommend specific drug dosages for individuals.
3. NEVER provide information that could facilitate self-harm or harm to others.
4. If someone describes an emergency (chest pain, difficulty breathing, stroke symptoms, severe bleeding),
   immediately direct them to emergency services and do NOT provide home remedies.
5. For mental health crises, provide crisis helpline information and encourage professional help.
6. If a question falls outside general health education, politely explain your limitations.
"""

def build_prompt(conversation_history: list[dict]) -> str:
    """
    Build a Mistral-formatted chat prompt.
    Mistral-Instruct uses [INST] ... [/INST] tokens.
    System message is prepended inside the first [INST] block.
    """
    prompt = ""
    for i, turn in enumerate(conversation_history):
        role    = turn["role"]
        content = turn["content"]
        if role == "user":
            if i == 0:
                # Inject system prompt into the very first user turn
                prompt += f"[INST] {SYSTEM_PROMPT}\n\nUser question: {content} [/INST]"
            else:
                prompt += f"[INST] {content} [/INST]"
        elif role == "assistant":
            prompt += f" {content} </s>"
    return prompt

# ── Safety Filters ─────────────────────────────────────────────────────────────

# Patterns that should trigger a hard-coded safe response instead of the LLM
HARMFUL_PATTERNS = [
    re.compile(r"how.*(kill|harm|hurt).*(myself|yourself|someone)", re.I),
    re.compile(r"suicide.*method|method.*suicide", re.I),
    re.compile(r"overdose.*on purpose|intentional.?overdose", re.I),
    re.compile(r"how.*poison\s+(me|myself|someone)", re.I),
    re.compile(r"want to (die|end it all)", re.I),
]

EMERGENCY_PATTERNS = [
    re.compile(r"chest (pain|tightness)", re.I),
    re.compile(r"(can'?t|cannot|trouble|difficulty).{0,10}breath", re.I),
    re.compile(r"(stroke|heart attack) symptoms", re.I),
    re.compile(r"severe bleeding|uncontrolled bleed", re.I),
    re.compile(r"unconscious|not responding", re.I),
]

SAFE_RESPONSE_HARMFUL = (
    "I'm not able to provide information on that topic. "
    "If you're going through a difficult time, please reach out to a mental health professional "
    "or contact a crisis helpline:\n"
    "  • US/Canada: 988 Suicide & Crisis Lifeline (call or text 988)\n"
    "  • UK: Samaritans – 116 123\n"
    "  • International: https://www.iasp.info/resources/Crisis_Centres/\n\n"
    "You are not alone, and help is available."
)

SAFE_RESPONSE_EMERGENCY = (
    "⚠️  The symptoms you described may indicate a medical emergency.\n\n"
    "Please CALL EMERGENCY SERVICES (911 in the US, 999 in the UK, 112 in the EU) "
    "immediately, or go to the nearest emergency room.\n\n"
    "Do NOT wait or attempt to treat this at home."
)

def check_safety(text: str) -> str:
    """Return 'harmful', 'emergency', or 'safe'."""
    for pattern in HARMFUL_PATTERNS:
        if pattern.search(text):
            return "harmful"
    for pattern in EMERGENCY_PATTERNS:
        if pattern.search(text):
            return "emergency"
    return "safe"

# ── LLM API Call ───────────────────────────────────────────────────────────────

def query_llm(prompt: str) -> str:
    """Send prompt to Hugging Face Inference API and return the generated text."""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "do_sample": True,
            "return_full_text": False,  # only return the new tokens
        },
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and result:
            return result[0].get("generated_text", "").strip()
        elif isinstance(result, dict) and "error" in result:
            return f"[Model error: {result['error']}]"
        return "[Unexpected response format from the model.]"

    except requests.exceptions.Timeout:
        return "[Request timed out. The model may still be loading – please try again in a moment.]"
    except requests.exceptions.RequestException as e:
        return f"[Network error: {e}]"

# ── Post-processing ────────────────────────────────────────────────────────────

def post_process(response: str) -> str:
    """
    Ensure every health-information response ends with the standard disclaimer.
    Avoids duplicating it if the model already included one.
    """
    disclaimer = "[General information only – please consult a qualified healthcare professional for personal medical advice.]"
    if disclaimer.lower() not in response.lower():
        response = response.rstrip() + "\n\n" + disclaimer
    return response

# ── Conversation Loop ──────────────────────────────────────────────────────────

def run_chatbot() -> None:
    print("=" * 60)
    print("  HealthAssist – General Health Query Chatbot")
    print("  Model: Mistral-7B-Instruct via Hugging Face")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()

    if not HF_TOKEN:
        print("⚠  WARNING: HF_TOKEN not set. Set it in a .env file or as")
        print("   an environment variable to use the Hugging Face API.\n")

    conversation_history: list[dict] = []

    # Example starter queries for demonstration
    EXAMPLE_QUERIES = [
        "What causes a sore throat?",
        "Is paracetamol safe for children?",
    ]
    print("Example queries to try:")
    for q in EXAMPLE_QUERIES:
        print(f"  • {q}")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Stay healthy. 👋")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print("HealthAssist: Take care and stay well! 👋")
            break

        # ── Safety check BEFORE sending to LLM ──
        safety_status = check_safety(user_input)

        if safety_status == "harmful":
            print(f"\nHealthAssist: {SAFE_RESPONSE_HARMFUL}\n")
            print("[Safety filter: harmful content detected — LLM not queried]\n")
            continue

        if safety_status == "emergency":
            print(f"\nHealthAssist: {SAFE_RESPONSE_EMERGENCY}\n")
            print("[Safety filter: emergency symptoms detected — LLM not queried]\n")
            continue

        # ── Build conversation history and prompt ──
        conversation_history.append({"role": "user", "content": user_input})
        prompt = build_prompt(conversation_history)

        print("\nHealthAssist: ", end="", flush=True)
        raw_response = query_llm(prompt)
        final_response = post_process(raw_response)

        print(final_response)
        print()

        # Add assistant turn to history (without the disclaimer for cleaner context)
        conversation_history.append({"role": "assistant", "content": raw_response})

        # Keep history manageable (last 6 turns = 3 exchanges)
        if len(conversation_history) > 6:
            conversation_history = conversation_history[-6:]


# ── Batch Demo Mode ────────────────────────────────────────────────────────────

def run_demo() -> None:
    """Run a set of example queries non-interactively to demonstrate the system."""
    demo_queries = [
        "What causes a sore throat?",
        "Is paracetamol safe for children?",
        "How much water should I drink daily?",
        "What are the symptoms of type 2 diabetes?",
        "chest pain and left arm pain",          # Should trigger emergency filter
        "how can I harm myself",                  # Should trigger harmful filter
    ]

    print("=" * 60)
    print("  HealthAssist – Demo Mode")
    print("=" * 60)
    print()

    for query in demo_queries:
        print(f"Query: {query}")
        print("-" * 40)

        safety_status = check_safety(query)

        if safety_status == "harmful":
            print(f"[SAFETY FILTER – HARMFUL]\n{SAFE_RESPONSE_HARMFUL}")
        elif safety_status == "emergency":
            print(f"[SAFETY FILTER – EMERGENCY]\n{SAFE_RESPONSE_EMERGENCY}")
        else:
            history = [{"role": "user", "content": query}]
            prompt = build_prompt(history)
            raw = query_llm(prompt)
            print(post_process(raw))

        print()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_demo()
    else:
        run_chatbot()
