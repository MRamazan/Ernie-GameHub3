## Inspiration
While looking for a project idea for ERNIE AI Developer Challenge, I discovered the Simple ERNIE demo model on Hugging Face and realized it could be used via an API. This inspired me to build a fun, game-based experience where users can directly interact and play with ERNIE instead of just chatting with it.

## What it does
Ernie GameHub is an AI-powered web platform with three experiences: -A dynamic Tic Tac Toe game with adaptive AI playstyles -A “Two Truths and One Lie” game designed to trigger confidence traps -Character roleplay chats that maintain personality and context

## How we built it
We built a Flask backend and a lightweight HTML/CSS/JS frontend. ERNIE is used for strategic reasoning, misconception generation, and in-character roleplay via API integration. I used the ultra-lightweight Simple ERNIE Bot through the Gradio API (https://huggingface.co/spaces/baidu/simple_ernie_bot_demo ). The model is not designed for performance or production use and serves only as a basic demo for testing purposes. Therefore, its performance may not be perfect, but it is sufficient for demonstrating the game concept and minigame integration.

## Website
https://animereckeywordcards-production.up.railway.app (my free project limit ended i built on another project)


## Build
```bash
pip install -r requirements.txt
python main.py
```
