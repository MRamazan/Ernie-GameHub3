from flask import Flask, render_template, request, jsonify, session
import sys
import pickle
from utils import *
from options import args
from models import model_factory

app = Flask(__name__)
app.secret_key = '1903bjk'


class AnimeRecommendationSystem:
    def __init__(self, checkpoint_path, dataset_path, animes_path, images_path):
        self.model = None
        self.dataset = None
        self.id_to_anime = {}
        self.id_to_url = {}
        self.checkpoint_path = checkpoint_path
        self.dataset_path = dataset_path
        self.animes_path = animes_path
        self.images_path = images_path
        self.load_model_and_data()

    def load_model_and_data(self):
        try:
            print("Loading model and data...")

            args.bert_max_len = 128

            dataset_path = Path(self.dataset_path)
            with dataset_path.open('rb') as f:
                self.dataset = pickle.load(f)

            with open(self.animes_path, "r", encoding="utf-8") as file:
                self.id_to_anime = json.load(file)

            try:
                with open(self.images_path, "r", encoding="utf-8") as file:
                    self.id_to_url = json.load(file)
                print(f"Loaded {len(self.id_to_url)} image URLs")
            except Exception as e:
                print(f"Warning: Could not load image URLs: {str(e)}")
                self.id_to_url = {}

            self.model = model_factory(args)
            self.load_checkpoint()

            print("Model loaded successfully!")

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def load_checkpoint(self):
        try:
            with open(self.checkpoint_path, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception as e:
            raise Exception(f"Failed to load checkpoint from {self.checkpoint_path}: {str(e)}")

    def get_all_animes(self):
        """Tüm anime listesini döndürür"""
        animes = []
        for k, v in self.id_to_anime.items():
            anime_name = v[0] if isinstance(v, list) and len(v) > 0 else str(v)
            animes.append((int(k), anime_name))
        animes.sort(key=lambda x: x[1])
        return animes

    def get_anime_image_url(self, anime_id):
        return self.id_to_url.get(str(anime_id), None)

    def get_recommendations(self, favorite_anime_ids, num_recommendations=40):
        try:
            if not favorite_anime_ids:
                return [], [], "Please add some favorite animes first!"

            smap = self.dataset
            inverted_smap = {v: k for k, v in smap.items()}

            converted_ids = []
            for anime_id in favorite_anime_ids:
                if anime_id in smap:
                    converted_ids.append(smap[anime_id])

            if not converted_ids:
                return [], [], "None of the selected animes are in the model vocabulary!"

            target_len = 128
            padded = converted_ids + [0] * (target_len - len(converted_ids))
            input_tensor = torch.tensor(padded, dtype=torch.long).unsqueeze(0)


            max_predictions = min(100, len(inverted_smap))

            with torch.no_grad():
                logits = self.model(input_tensor)
                last_logits = logits[:, -1, :]
                top_scores, top_indices = torch.topk(last_logits, k=max_predictions, dim=1)

      
            recommendations = []
            scores = []

            for idx, score in zip(top_indices.numpy()[0], top_scores.detach().numpy()[0]):
                if idx in inverted_smap:
                    anime_id = inverted_smap[idx]
                 
                    if anime_id in favorite_anime_ids:
                        continue
                    if str(anime_id) in self.id_to_anime:
                        anime_data = self.id_to_anime[str(anime_id)]
                     
                        anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(
                            anime_data)

                     
                        image_url = self.get_anime_image_url(anime_id)

                        recommendations.append({
                            'id': anime_id,
                            'name': anime_name,
                            'score': float(score),
                            'image_url': image_url
                        })
                        scores.append(float(score))

                        if len(recommendations) >= num_recommendations:
                            break

            return recommendations, scores, f"Found {len(recommendations)} recommendations!"

        except Exception as e:
            return [], [], f"Error during prediction: {str(e)}"


recommendation_system = None


@app.route('/')
def index():
    """Ana sayfa"""
    if recommendation_system is None:
        return render_template('error.html',
                               error="Recommendation system not initialized. Please check server logs.")

    animes = recommendation_system.get_all_animes()
    return render_template('index.html', animes=animes)


@app.route('/api/search_animes')
def search_animes():
    """Anime arama API'si"""
    query = request.args.get('q', '').lower()
    animes = []

    for k, v in recommendation_system.id_to_anime.items():
        anime_names = v if isinstance(v, list) else [v]
     
        match_found = False
        for name in anime_names:
            if query in name.lower():
                match_found = True
                break

        if not query or match_found:
            main_name = anime_names[0] if anime_names else "Unknown"
            animes.append((int(k), main_name))

    animes.sort(key=lambda x: x[1])
    return jsonify(animes)


@app.route('/api/add_favorite', methods=['POST'])
def add_favorite():
    """Favorilere anime ekleme"""
    if 'favorites' not in session:
        session['favorites'] = []

    data = request.get_json()
    anime_id = int(data['anime_id'])
    anime_name = data['anime_name']

    if anime_id not in session['favorites']:
        session['favorites'].append(anime_id)
        session.modified = True
        return jsonify({'success': True, 'message': f"'{anime_name}' added to favorites!"})
    else:
        return jsonify({'success': False, 'message': f"'{anime_name}' is already in favorites!"})


@app.route('/api/remove_favorite', methods=['POST'])
def remove_favorite():
    """Favorilerden anime çıkarma"""
    if 'favorites' not in session:
        session['favorites'] = []

    data = request.get_json()
    anime_id = int(data['anime_id'])
    anime_name = data['anime_name']

    if anime_id in session['favorites']:
        session['favorites'].remove(anime_id)
        session.modified = True
        return jsonify({'success': True, 'message': f"'{anime_name}' removed from favorites!"})
    else:
        return jsonify({'success': False, 'message': f"'{anime_name}' not found in favorites!"})


@app.route('/api/clear_favorites', methods=['POST'])
def clear_favorites():
    """Tüm favorileri temizle"""
    session['favorites'] = []
    session.modified = True
    return jsonify({'success': True, 'message': 'All favorites cleared!'})


@app.route('/api/get_favorites')
def get_favorites():
    """Favori anime listesini döndür"""
    if 'favorites' not in session:
        session['favorites'] = []

    favorite_animes = []
    for anime_id in session['favorites']:
        if str(anime_id) in recommendation_system.id_to_anime:
            anime_data = recommendation_system.id_to_anime[str(anime_id)]
      
            anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)
            favorite_animes.append({'id': anime_id, 'name': anime_name})

    return jsonify(favorite_animes)


@app.route('/api/get_recommendations', methods=['POST'])
def get_recommendations():
    """Öneri alma"""
    if 'favorites' not in session or not session['favorites']:
        return jsonify({'success': False, 'message': 'Please add some favorite animes first!'})

    recommendations, scores, message = recommendation_system.get_recommendations(session['favorites'])

    if recommendations:
        return jsonify({
            'success': True,
            'recommendations': recommendations,
            'message': message
        })
    else:
        return jsonify({'success': False, 'message': message})


def main():
    global recommendation_system

    args.num_items = 12689

    import gdown
    import os

    file_ids = {
        "1C6mdjblhiWGhRgbIk5DP2XCc4ElS9x8p": "pretrained_bert.pth",
        "1U42cFrdLFT8NVNikT9C5SD9aAux7a5U2": "animes.json",
        "1s-8FM1Wi2wOWJ9cstvm-O1_6XculTcTG": "dataset.pkl",
        "1SOm1llcTKfhr-RTHC0dhaZ4AfWPs8wRx": "id_to_url.json"
    }

    def download_from_gdrive(file_id, output_path):
        """Google Drive'dan dosya indir"""
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            print(f"Downloading: {file_id}")
            gdown.download(url, output_path, quiet=False)
            print(f"Downloaded: {output_path}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    for key, value in file_ids.items():
        if os.path.isfile(value):
            continue
        download_from_gdrive(key, value)

    try:

        images_path = "id_to_url.json"
        if not os.path.exists(images_path):
            print(f"Warning: {images_path} not found. Images will not be displayed.")

        recommendation_system = AnimeRecommendationSystem(
            "pretrained_bert.pth",
            "dataset.pkl",
            "animes.json",
            images_path
        )
        print("Recommendation system initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize recommendation system: {e}")
        sys.exit(1)

    app.run(debug=True, host='0.0.0.0', port=5000)


from google import genai
from google.genai import types
import json
import time
from functools import lru_cache




@lru_cache(maxsize=1000)
def get_anime_keywords_cached(anime_name: str) -> dict:
    try:
        print(f"[INFO] Getting keywords for: {anime_name}")

        model_config = types.GenerateContentConfig(
            temperature=1.0,
            top_p=0.95,
        )
        client = genai.Client(api_key="API_KEY")

        prompt = f"""
        Write 4 key words for and their japanese and romaji translations '{anime_name}' anime. 
        No chatting, only write words in json dictionary format. key should be english of the word and value should be japanese of the word, romaji of the word in paranthesis should be next to japanese title. Both key and value must be strings"""

        prompt2 = """
        EXAMPLE OUTPUT:
        {
        "Reincarnation": "転生 (Tensei)"
        "Magic": "魔法 (Mahō)"
        "Adventure": "冒険 (Bōken)"
        "Family": "家族 (Kazoku)"
        }
        """

        prompt += prompt2

        response = client.models.generate_content(
            model='gemini-2.0-flash',
            config=model_config,
            contents=prompt
        )

        response_text = response.text
        keywords_dict = eval(response_text.replace("json", "").replace("```", ""))
        return keywords_dict

    except Exception as e:
        print(f"Error getting keywords for {anime_name}: {str(e)}")
     
        return {
            "Anime": "アニメ (Anime)",
            "Story": "物語 (Monogatari)",
            "Character": "キャラクター (Kyarakutā)",
            "Adventure": "冒険 (Bōken)",
            "Emotion": "感情 (Kanjō)"
        }


@app.route('/api/get_keywords/<int:anime_id>')
def get_keywords(anime_id):
    """Anime için anahtar kelimeleri döndür"""
    try:
        if str(anime_id) not in recommendation_system.id_to_anime:
            return jsonify({'success': False, 'message': 'Anime not found'})

        anime_data = recommendation_system.id_to_anime[str(anime_id)]
        anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)

        keywords = get_anime_keywords_cached(anime_name)

        return jsonify({
            'success': True,
            'keywords': keywords,
            'anime_name': anime_name
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error getting keywords: {str(e)}'
        })


keyword_requests = {}


@app.route('/api/get_keywords_batch', methods=['POST'])
def get_keywords_batch():
    try:
        data = request.get_json()
        anime_ids = data.get('anime_ids', [])

        if len(anime_ids) > 10:  
            return jsonify({'success': False, 'message': 'Too many requests at once'})

        results = {}

        for anime_id in anime_ids:
            if str(anime_id) in recommendation_system.id_to_anime:
                anime_data = recommendation_system.id_to_anime[str(anime_id)]
                anime_name = anime_data[0] if isinstance(anime_data, list) and len(anime_data) > 0 else str(anime_data)

                try:
                    keywords = get_anime_keywords_cached(anime_name)
                    results[anime_id] = {
                        'success': True,
                        'keywords': keywords,
                        'anime_name': anime_name
                    }
                except Exception as e:
                    results[anime_id] = {
                        'success': False,
                        'message': str(e)
                    }

                time.sleep(0.5)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error in batch request: {str(e)}'
        })


if __name__ == "__main__":
    main()
