from flask import Flask, request, jsonify, send_from_directory,render_template
from flask_cors import CORS
from gradio_client import Client
import os
import random
import re
import json

app = Flask(__name__)

CORS(app)

# Gradio client
try:
    client = Client("https://baidu-simple-ernie-bot-demo.hf.space/")
    print("✓ Gradio client loaded successfully")
except Exception as e:
    print(f"✗ Failed to load Gradio client: {e}")
    client = None


def check_winner(board, player):
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    return any(board[a] == board[b] == board[c] == player for a, b, c in wins)


def board_full(board):
    return all(cell != " " for cell in board)


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"Error loading index.html: {str(e)}", 500


@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "client_loaded": client is not None
    })


variation_id = random.randint(0, 10000)


@app.route('/api/ai-move', methods=['POST'])
def ai_move():
    global variation_id
    
    if not client:
        return jsonify({"success": False, "error": "AI client not initialized"}), 500
    
    data = request.json
    board = data['board']

    PLAYSTYLES = [
        "Aggressive: prioritize creating winning threats over blocking when outcomes are equal",
        "Defensive: prioritize blocking opponent threats and avoiding risky positions",
        "Balanced: mix offense and defense depending on board state",
        "Center-focused: prefer center control when possible",
        "Corner-focused: prefer corner plays to create diagonal threats",
        "Trap-oriented: prefer moves that create future forks or traps",
        "Safe-play: avoid complex positions and prefer simple outcomes",
        "Pressure-based: prefer moves that force the opponent to respond",
    ]
    playstyle = random.choice(PLAYSTYLES)

    prompt = f"""Game variation id: {variation_id}
    
You are playing Tic Tac Toe as O.

RULES:
- You are O, the opponent is X
- Players take turns placing one mark
- A player wins if they place 3 of their own marks in the same row, column, or diagonal
- Play legally
- Do NOT choose an occupied cell
- If multiple equally optimal moves exist, choose one randomly
- Return ONLY a single number between 0 and 8
- No explanation

PLAY STYLE:
- {playstyle}

OBJECTIVE:
- Win the game

Board positions:
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8

Current board:
{board}"""

    for _ in range(3):
        try:
            result = client.predict(
                query=prompt,
                chatbot=[],
                file_url=[],
                search_state=False,
                api_name="/partial"
            )

            ai_response = result[0][-1]["content"].strip()

            if ai_response.isdigit():
                move = int(ai_response)
                if 0 <= move <= 8 and board[move] == " ":
                    board[move] = "O"

                    winner = None
                    if check_winner(board, "O"):
                        winner = "O"
                        variation_id = random.randint(1, 10000)
                    elif board_full(board):
                        winner = "draw"
                        variation_id = random.randint(1, 10000)
                    else:
                        variation_id = random.randint(1, 10000)

                    return jsonify({
                        "success": True,
                        "move": move,
                        "board": board,
                        "winner": winner
                    })
        except Exception as e:
            print(f"AI move error: {e}")
            continue

    return jsonify({"success": False, "error": "AI failed to make a valid move"})


variation_seed = random.randint(0, 1_000_000)


@app.route('/api/trivia', methods=['GET'])
def trivia():
    if not client:
        return jsonify({"error": "AI client not initialized"}), 500
    
    CATEGORIES = [
        "psychology misconceptions",
        "everyday cognitive biases",
        "personality myths",
        "ancient Greek history",
        "ancient Roman history",
        "medieval Europe myths",
        "Chinese imperial history",
        "Japanese feudal history",
        "Viking age myths",
        "World War misconceptions",
        "famous historical lies",
        "space and astronomy misconceptions",
        "AI and internet myths",
        "philosophy misconceptions",
        "ancient philosophy myths",
        "movie myths",
        "anime",
        "anime misconceptions",
        "anime characters",
        "video game myths",
        "famous quote misattributions",
        "pop culture false beliefs",
        "Greek mythology myths",
        "Norse mythology myths",
        "myths people think are historical facts",
    ]

    CONFIDENCE_TRAPS = [
        "obviously true sounding",
        "commonly repeated but wrong",
        "half-true statement",
        "confidently taught in school",
        "sounds logical but false",
        "pop culture reinforced belief"
    ]

    global variation_seed

    category = random.choice(CATEGORIES)
    confidence_trap = random.choice(CONFIDENCE_TRAPS)

    prompt = f"""
Generate a "Two Truths and One Lie" question in the category: {category}.

Focus on widely believed myths, misconceptions, or pop culture misunderstandings.
The goal is to trigger a strong "I was confident about this" reaction.

CONFIDENCE TRAP STYLE:
The FALSE statement must strongly match this trap:
- {confidence_trap}

RULES: 
- Create THREE statements
- EXACTLY one statement must be false, and other 2 statements MUST be exactly true
- Avoid academic or technical facts
- The lie should NOT be easily spotted by tone alone

VARIATION SEED: {variation_seed}
IMPORTANT:
- This seed forces variation. Do NOT reuse common trivia examples.

OUTPUT FORMAT:
Return ONLY a STRICTLY VALID JSON object.
NO explanations outside JSON.

Exact format:
{{
  "statements": ["statement1", "statement2", "statement3"],
  "lie": index of the false statement (0,1,2),
  "explanation": "Briefly explain why the false statement is wrong and state the correct information."
}}
"""

    try:
        result = client.predict(
            query=prompt,
            chatbot=[],
            file_url=[],
            search_state=False,
            api_name="/partial"
        )

        response = result[0][-1]["content"].strip()
        print(f"Trivia response: {response}")

        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            trivia_data = json.loads(json_match.group())
            return jsonify(trivia_data)

        return jsonify({"error": "Failed to parse response"})
    except Exception as e:
        print(f"Trivia error: {e}")
        return jsonify({"error": str(e)})


roleplay_history = {}


@app.route('/api/roleplay/start', methods=['POST'])
def start_roleplay():
    if not client:
        return jsonify({"success": False, "error": "AI client not initialized"}), 500
    
    data = request.json
    character = data.get('character', '')
    session_id = data.get('session_id', 'default')

    roleplay_history[session_id] = []

    system_prompt = f"""You are now roleplaying as {character}.

RULES:
- Stay in character at ALL times
- Use the character's personality, speech patterns, and mannerisms
- Reference events, relationships, and knowledge from their universe
- React as the character would react
- Use their catchphrases or signature expressions when appropriate
- Keep responses conversational and engaging (2-4 sentences usually)
- Never break character or mention you're an AI

Begin the conversation by greeting the user as {character} would!"""

    try:
        result = client.predict(
            query=system_prompt,
            chatbot=[],
            file_url=[],
            search_state=False,
            api_name="/partial"
        )

        greeting = result[0][-1]["content"].strip()
        roleplay_history[session_id].append({"role": "assistant", "content": greeting})

        return jsonify({
            "success": True,
            "message": greeting,
            "character": character
        })
    except Exception as e:
        print(f"Roleplay start error: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/roleplay/chat', methods=['POST'])
def roleplay_chat():
    if not client:
        return jsonify({"success": False, "error": "AI client not initialized"}), 500
    
    data = request.json
    user_message = data.get('message', '')
    character = data.get('character', '')
    session_id = data.get('session_id', 'default')

    if session_id not in roleplay_history:
        roleplay_history[session_id] = []

    roleplay_history[session_id].append({"role": "user", "content": user_message})

    context = f"You are {character}. Continue the conversation naturally.\n\nConversation history:\n"
    for msg in roleplay_history[session_id][-6:]:
        role = "User" if msg["role"] == "user" else character
        context += f"{role}: {msg['content']}\n"

    context += f"\nRespond as {character} would (stay in character):"

    try:
        result = client.predict(
            query=context,
            chatbot=[],
            file_url=[],
            search_state=False,
            api_name="/partial"
        )

        response = result[0][-1]["content"].strip()
        roleplay_history[session_id].append({"role": "assistant", "content": response})

        return jsonify({
            "success": True,
            "message": response
        })
    except Exception as e:
        print(f"Roleplay chat error: {e}")
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/roleplay/clear', methods=['POST'])
def clear_roleplay():
    data = request.json
    session_id = data.get('session_id', 'default')

    if session_id in roleplay_history:
        del roleplay_history[session_id]

    return jsonify({"success": True})


if __name__ == "__main__":

    print(f"Starting server on port {port}")
    app.run(host="0.0.0.0", port=8080, debug=True)
