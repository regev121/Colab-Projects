import pandas as pd
import numpy as np
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.spatial.distance import cosine
import joblib
import string


def preprocess_sentences(text: str) -> list:
    """Preprocess sentences by converting to lowercase, removing punctuation, and tokenizing"""
    text = text.lower()
    sentences = sent_tokenize(text)
    cleaned_sentences = []

    for sentence in sentences:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        tokens = sentence.split()
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        cleaned_sentences.append(" ".join(tokens))

    return cleaned_sentences


def predict_sentiment(review: str, genre: str) -> dict:
    """Predict sentiment for a review and genre"""
    sia = SentimentIntensityAnalyzer()
    cat_dic = {'visuals': [], 'soundtrack': [], 'gameplay': []}

    for r in preprocess_sentences(review):
        sen = sia.polarity_scores(r)['compound']
        for category in keywords.keys():
            if any(word in r.lower().split() for word in keywords[category]):
                cat_dic[category].append(sen)

    for cal, cal_sent in cat_dic.items():
        if cal_sent:
            cat_dic[cal] = np.mean(cal_sent)
        else:
            cat_dic[cal] = 0

    return cat_dic


def predict_game_features(review: str, genre: str) -> pd.DataFrame:
    """Predict game features from review and genre"""
    new_game_features = pd.DataFrame(columns=['review', 'Action', 'Adventure', 'Casual', 'Early Access', 'Free to Play', 'Indie',
                                             'Massively Multiplayer', 'Nudity', 'RPG', 'Racing', 'Simulation', 'Sports', 'Strategy'])
    new_game_features.loc[0, 'review'] = review
    str_list = genre.split(',')

    for genre in str_list:
        genre = genre.strip()
        if genre in new_game_features.columns:
            new_game_features.loc[0, genre] = 1
    new_game_features = new_game_features.fillna(0)

    new_game_features['processed_review'] = new_game_features['review'].apply(preprocess_sentences)

    sentiment_features = predict_sentiment(review, genre)
    new_game_features['visuals'] = sentiment_features['visuals']
    new_game_features['soundtrack'] = sentiment_features['soundtrack']
    new_game_features['gameplay'] = sentiment_features['gameplay']

    return new_game_features


def load_rfc_model() -> object:
    """Load RFC model from file"""
    try:
        return joblib.load('rfcmodel.joblib')
    except Exception as e:
        print(f"Error loading RFC model: {e}")
        return None


def PREDICT(review: str, genre: str) -> bool:
    """Predict game recommendation based on review and genre"""
    rfc_model = load_rfc_model()
    if rfc_model is None:
        return False

    new_game_features = predict_game_features(review, genre)
    x = new_game_features[['visuals', 'soundtrack', 'gameplay']]
    prediction = rfc_model.predict(x)

    return prediction == 1


keywords = {
    'gameplay': ['Immersive', 'Dynamic', 'Strategic', 'Responsive', 'Intuitive', 'Challenging', 'Seamless', 'Adaptive', 'Engaging', 'Tactical', 'Fluid', 'Puzzling', 'Navigable', 'Interactive', 'Strategic', 'Reactive', 'Versatile', 'Progressive', 'Innovative', 'Satisfying','story', 'missions', 'characters', 'shooting', 'controls', 'boss fights', 'gameplay', 'setting', 'exploration', 'scavenging', 'weapons', 'player interaction', 'attack', 'sandbox', 'exploring', 'survival', 'combat', 'abilities', 'evolving', 'weapons', 'resource production', 'mechanics', 'learning curve', 'decision-making', 'consequence', 'frustrating', 'difficulty levels', 'multiplayer', 'online', 'single-player', 'campaign', 'playthrough', 'terraforming', 'colonization', 'space', 'outposts', 'domes', 'rockets', 'balancing', 'resource distribution', 'disasters', 'replayability', 'achievements', 'challenges', 'strategy', 'simulations'],
    'soundtrack': ['Melodic', 'Atmospheric', 'Harmonious', 'Cinematic', 'Evocative', 'Enchanting', 'Uplifting', 'Lyrical', 'Emotional', 'Energetic', 'Ambient', 'Captivating', 'Haunting', 'Majestic', 'Rhythmic', 'Soulful', 'Orchestral', 'Nostalgic', 'Mesmerizing', 'music', 'soundtrack', 'themes', 'sound', 'relaxing', 'plays', 'various', 'psychopath', 'catchy', 'beat', 'sequence', 'type', 'ear', 'credits', 'catchy', 'cinematic', 'immersive', 'atmospheric', 'emotional', 'orchestral', 'repetitive'],
    'visuals': ['Photorealistic', 'Vibrant', 'Aesthetic', 'Cinematic', 'Crisp', 'Stunning', 'Immersive', 'Detailed', 'Sleek', 'Striking', 'Polished', 'Colorful', 'Realistic', 'Atmospheric', 'Dynamic', 'Vivid', 'Clean', 'Surreal', 'Sharp', 'Futuristic', 'discovery', 'visual', 'elements', 'textures', 'effects', 'environment', 'atmosphere', 'surface', 'atmospheric', 'representation', 'cracks', 'facade', 'creative', 'innovative', 'solutions', 'improving', 'enhancing', 'graphics', 'visuals', 'improvements']
}

RFCModel = load_rfc_model()