import streamlit as st
import random
import json
import asyncio
import httpx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Configuration and Setup ---
st.set_page_config(page_title="Lateral Thinking Puzzles", page_icon="🕵️‍♀️")

# --- Function to Load Puzzles ---
def load_puzzles(filepath="Project/Project/puzzles.json"):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Error loading puzzles: {e}")
        return []

# Load all puzzles
all_puzzles = load_puzzles()
if not all_puzzles:
    st.stop()

# --- Session State Management ---
if "current_puzzle_index" not in st.session_state:
    st.session_state["current_puzzle_index"] = random.randint(0, len(all_puzzles) - 1)
if "show_answer" not in st.session_state:
    st.session_state["show_answer"] = False
if "show_hint" not in st.session_state:
    st.session_state["show_hint"] = False
if "score" not in st.session_state:
    st.session_state["score"] = 0
if "ai_response" not in st.session_state:
    st.session_state["ai_response"] = ""
if "user_question_input" not in st.session_state:
    st.session_state["user_question_input"] = ""
if "user_guess_input" not in st.session_state:
    st.session_state["user_guess_input"] = ""
if "correct_guess" not in st.session_state:
    st.session_state["correct_guess"] = False
if "initialized" not in st.session_state:
    st.session_state["initialized"] = False
if "sentence_model" not in st.session_state:
    st.session_state["sentence_model"] = SentenceTransformer('paraphrase-MiniLM-L6-v2')
if "clear_score_pressed" not in st.session_state:
    st.session_state["clear_score_pressed"] = False
if "new_puzzle_pressed" not in st.session_state:
    st.session_state["new_puzzle_pressed"] = False
if "selected_puzzle_index" not in st.session_state:
    st.session_state["selected_puzzle_index"] = None
if "user_question_count" in st.session_state:
    del st.session_state["user_question_count"]

# --- Helper Function to Reset Puzzle ---
def reset_puzzle():
    st.session_state["current_puzzle_index"] = random.randint(0, len(all_puzzles) - 1)
    st.session_state["show_answer"] = False
    st.session_state["show_hint"] = False
    st.session_state["ai_response"] = ""
    st.session_state["correct_guess"] = False
    st.session_state["user_guess_input"] = ""
    st.session_state["user_question_input"] = ""
    st.session_state["new_puzzle_pressed"] = True # Indicate that a new puzzle was requested
    st.session_state["selected_puzzle_index"] = None

# --- Function to Clear Score ---
def clear_score():
    st.session_state["score"] = 0
    st.session_state["clear_score_pressed"] = True

# --- Function to Set Specific Puzzle ---
def set_puzzle(puzzle_index):
    if 0 <= puzzle_index < len(all_puzzles):
        st.session_state["current_puzzle_index"] = puzzle_index
        st.session_state["show_answer"] = False
        st.session_state["show_hint"] = False
        st.session_state["ai_response"] = ""
        st.session_state["correct_guess"] = False
        st.session_state["user_guess_input"] = ""
        st.session_state["user_question_input"] = ""
        st.session_state["selected_puzzle_index"] = puzzle_index
    else:
        st.error(f"Invalid puzzle number: {puzzle_index + 1}")

# --- AI Question Answering Function ---
async def get_ai_yes_no_answer(puzzle_text, puzzle_answer, user_question):
    prompt = f"""Given the lateral thinking puzzle: "{puzzle_text}" and its solution: "{puzzle_answer}", answer the question with only 'yes' or 'no'. If unsure, infer the most likely answer. Question: "{user_question}" """
    headers = {'Content-Type': 'application/json'}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    api_key = st.secrets.get("API_KEY")
    if not api_key:
        return "Error: Gemini API Key not found."
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(api_url, headers=headers, json=payload)
            response.raise_for_status()
            if response.json().get("candidates") and response.json()["candidates"][0].get("content").get("parts"):
                return response.json()["candidates"][0]["content"]["parts"][0].get("text", "").strip().lower()
            return "AI response was not clear."
    except httpx.HTTPError as e:
        return f"AI request failed: {e}"

# --- Function to Get Optimized Hints Regarding the Answer ---
def get_optimized_hint(puzzle):
    if "hints" in puzzle and puzzle["hints"]:
        return random.choice(puzzle["hints"])
    else:
        answer_parts = [word.lower() for word in puzzle["answer"].split() if len(word) > 2 and word.lower() not in st.session_state.get("common_words", [])]
        if answer_parts:
            return f"Consider the role of '{random.choice(answer_parts).capitalize()}' 🤔."
        else:
            return "Think about a key element or action in the scenario 🤔."

# --- Load Common Words (for hint optimization) ---
@st.cache_data
def load_common_words():
    return set(["the", "a", "is", "was", "are", "were", "and", "but", "or", "in", "on", "at", "to", "from", "by", "with", "of", "it", "that", "this", "he", "she", "they", "them", "his", "her", "their"])
st.session_state["common_words"] = load_common_words()

# --- Function to extract keywords from text ---
def extract_keywords(text):
    return re.findall(r'\b\w+\b', text.lower())

# --- Function to check if the guess is similar to the answer using sentence embeddings and keyword overlap ---
def is_similar(guess, answer, similarity_threshold=0.8, keyword_threshold=0.6):
    guess_lower = guess.strip().lower()
    answer_lower = answer.lower()

    embeddings = st.session_state["sentence_model"].encode([guess_lower, answer_lower])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    if similarity >= similarity_threshold:
        return True
    else:
        # Check for keyword overlap as a secondary measure
        guess_keywords = extract_keywords(guess_lower)
        answer_keywords = extract_keywords(answer_lower)
        if answer_keywords:
            common_keywords_ratio = len(set(guess_keywords) & set(answer_keywords)) / len(answer_keywords)
            return common_keywords_ratio >= keyword_threshold
        return False

# --- Get the Current Puzzle ---
current_puzzle = all_puzzles[st.session_state["current_puzzle_index"]]

# --- User Interface (UI) Layout ---
st.title("🕵️‍♀️ Lateral Thinking Puzzles Game")

with st.expander("How to Play"):
    st.markdown("""
    Lateral thinking puzzles are riddles with peculiar scenarios. Solve them by asking yes/no questions to the AI.
    * **Read the puzzle.** 🧐
    * **Ask yes/no questions** using the 'Ask AI' box. 🗣️
    * **Formulate your theory** in the 'What's your theory?' box. 💡
    * **Submit your guess.** ✅
    * **Show Answer** if you're stuck. 🔑
    * **Get a Hint** related to the answer. ❓ 
                 """)

st.markdown("---")

# --- Puzzle Selection ---
st.sidebar.header("Choose Puzzle 🧩")
puzzle_options = [f"{i+1}: {p['title']}" for i, p in enumerate(all_puzzles)]
selected_puzzle_option = st.sidebar.selectbox("Select a puzzle:", puzzle_options)
selected_puzzle_index = puzzle_options.index(selected_puzzle_option)
if selected_puzzle_index != st.session_state.get("selected_puzzle_index"):
    set_puzzle(selected_puzzle_index)

st.subheader(f"Puzzle: {current_puzzle['title']}")
st.markdown(f"**Scenario:** {current_puzzle['puzzle']}")

st.markdown("---")

st.subheader("Ask the AI (Yes/No Questions) 🤖")
user_question = st.text_input("Ask a yes/no question:", key="user_question_input", placeholder="e.g., Was anyone hurt?")

if st.button("Ask AI", use_container_width=True):
    if user_question:
        with st.spinner("AI is thinking... 🧠"):
            st.session_state["ai_response"] = asyncio.run(
                get_ai_yes_no_answer(
                    current_puzzle["puzzle"],
                    current_puzzle["answer"],
                    user_question
                )
            )
    else:
        st.warning("Please ask a question.")

if st.session_state["ai_response"]:
    st.info(f"AI's Answer: **{st.session_state['ai_response']}**")

st.markdown("---")

st.subheader("Your Solution 💡")
user_guess = st.text_input("What's your theory?", key="user_guess_input", placeholder="Type your answer here...")

col1, col2 = st.columns(2)

with col1:
    if st.button("Submit Guess ✅", use_container_width=True):
        if user_guess:
            if is_similar(user_guess.strip(), current_puzzle["answer"]):
                st.session_state["correct_guess"] = True
                st.session_state["score"] += 1
                st.session_state["show_answer"] = True # Automatically show the answer
            else:
                st.error("🤔 Not quite. Try asking more questions or get a hint.")
        else:
            st.warning("Please type your guess.")

with col2:
    if st.button("Show Answer 🔑", use_container_width=True):
        st.session_state["show_answer"] = not st.session_state["show_answer"]

st.markdown("---")

# --- Display "Correct" Message and Automatically Show Answer ---
if st.session_state["correct_guess"]:
    st.success("🎉 Correct!")
    st.info(f"**The Answer Is:** {current_puzzle['answer']}")
    st.markdown("---") # Add a separator after the answer
    # Optionally reset the puzzle after a correct guess
    st.button("Next Puzzle ➡️", on_click=reset_puzzle, use_container_width=True)
else:
    # Only display the answer if the "Show Answer" button was pressed
    if st.session_state["show_answer"]:
        st.info(f"**The Answer Is:** {current_puzzle['answer']}")
        st.markdown("---") # Add a separator after the answer

st.subheader("Need a Hint? 🤔")
if st.button("Get a Hint 💡", use_container_width=True):
    st.session_state["show_hint"] = True

if st.session_state["show_hint"]:
    st.info(f"**Hint:** {get_optimized_hint(current_puzzle)}")

st.markdown("---")

st.sidebar.metric("Solved", st.session_state["score"])
st.sidebar.markdown("---")
st.sidebar.button("New Random Puzzle 🎲", on_click=reset_puzzle, use_container_width=True, key="new_puzzle_button")
st.sidebar.button("Clear Score 🧹", on_click=clear_score, use_container_width=True, key="clear_score_button")
if st.session_state.get("clear_score_pressed"):
    st.sidebar.success("Score cleared!")
    st.session_state["clear_score_pressed"] = False

# This ensures the initial puzzle is loaded
if not st.session_state["initialized"]:
    st.session_state["initialized"] = True

# No need to call reset_puzzle() unconditionally here anymore.
# It's only called by the "New Random Puzzle" button or after a correct guess (optional).