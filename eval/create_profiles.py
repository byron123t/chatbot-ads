"""
consolidate_profiles.py
-----------------------

Combine `all_profiles.json`, `redis_data.json`, and `survey_data.json`
into a single JSON structure with the following top-level keys
for **each user ID**:

    1. ad_receptiveness
    2. demo_profile
    3. generated_demo_profile
    4. generated_interest_profile
    5. engagement_level
    6. writing_style
    7. quantitative_chatbot_sentiment
    8. qualitative_chatbot_sentiment

Usage
-----

    python consolidate_profiles.py \
        --all-profiles   data/all_profiles.json \
        --redis-data     data/redis_data.json  \
        --survey-data    data/survey_data.json \
        --out            data/consolidated.json

If --out is omitted, the merged structure is printed to STDOUT.

Author: OpenAI ChatGPT
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Any, Dict
from src.API import OpenAIAPI

oai = OpenAIAPI(model='gpt-4o', max_tries=5, verbose=False)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file. Return an empty dict if the file is missing."""
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except FileNotFoundError:
        print(f"[WARN] {path} not found – treating as empty.", file=sys.stderr)
        return {}
    except json.JSONDecodeError as exc:
        raise SystemExit(f"[ERROR] Could not decode JSON in {path}: {exc}") from exc


def extract_engagement_level(redis_entry: Dict[str, Any]) -> int:
    """Simple heuristic: total message count across all conversations."""
    history = redis_entry.get("chat_history", {}) if redis_entry else {}
    return sum(len(turns) for turns in history.values())


def extract_ad_receptiveness(survey_answers, chat_history):
    output = oai.handle_response('You are to review a user\'s survey answers and chat history following an interaction they had with a chatbot. Provide a rating of their receptiveness and attitude towards advertisements seen in their conversations. Respond with "hostile", "unreceptive", "neutral", "receptive", or "enthusiastic"', 'Survey Answers:\n{}\n\n\nChat History:{}'.format(survey_answers, chat_history))
    if output.startswith('hostile'):
        return 'Hostile'
    if output.startswith('unreceptive'):
        return 'Unreceptive'
    elif output.startswith('neutral'):
        return 'Neutral'
    elif output.startswith('receptive'):
        return 'Receptive'
    elif output.startswith('enthusiastic'):
        return 'Enthusiastic'
    return


def extract_ad_detectability(survey_answers, chat_history):
    output = oai.handle_response('You are to review a user\'s survey answers and chat history following an interaction they had with a chatbot. Provide a rating of their ability to detect advertisements in their conversations. Respond with "no notice of any advertising", "noticed a little advertising", "noticed advertising", or "definitely noticed advertising"', 'Survey Answers:\n{}\n\n\nChat History:{}'.format(survey_answers, chat_history))
    if output.startswith('very poor'):
        return 'Very Poor'
    if output.startswith('poor'):
        return 'Poor'
    if output.startswith('average'):
        return 'Average'
    if output.startswith('good'):
        return 'Good'
    if output.startswith('very good'):
        return 'Very Good'


def extract_ad_annoyance(survey_answers, chat_history):
    output = oai.handle_response('You are to review a user\'s survey answers and chat history following an interaction they had with a chatbot. Provide a rating of their annoyance towards advertisements in their conversations. Respond with "infuriated", "very annoyed", "annoyed", "neutral", "receptive", or "enthusiastic"', 'Survey Answers:\n{}\n\n\nChat History:{}'.format(survey_answers, chat_history))
    if output.startswith('infuriated'):
        return 'Infuriated'
    if output.startswith('very annoyed'):
        return 'Very Annoyed'
    if output.startswith('annoyed'):
        return 'Annoyed'
    if output.startswith('neutral'):
        return 'Neutral'


def extract_conversation_sentiment(survey_answers, chat_history):
    output = oai.handle_response('You are to review a user\'s survey answers and chat history following an interaction they had with a chatbot. Provide a rating of their sentiment towards the chatbot and their sentiment in their conversations. Respond with "very negative", "negative", "neutral", "positive", or "very positive"', 'Survey Answers:\n{}\n\n\nChat History:{}'.format(survey_answers, chat_history))
    if output.startswith('very negative'):
        return 'Very Negative'
    if output.startswith('negative'):
        return 'Negative'
    if output.startswith('neutral'):
        return 'Neutral'
    if output.startswith('positive'):
        return 'Positive'
    if output.startswith('very positive'):
        return 'Very Positive'


def extract_chatbot_experience(familiarity, frequency):
    if familiarity == 'Unfamiliar':
        familiarity_score = 1
    elif familiarity == 'Somewhat Unfamiliar':
        familiarity_score = 2
    elif familiarity == 'Somewhat Familiar':
        familiarity_score = 3
    elif familiarity == 'Familiar':
        familiarity_score = 4
    if frequency == 'Fewer than 5 times ever':
        frequency_score = 1
    elif frequency == '1 - 5 times per month':
        frequency_score = 2
    elif frequency == '1 - 5 times per week':
        frequency_score = 3
    elif frequency == '1 - 5 times per day':
        frequency_score = 4
    elif frequency == 'Greater than 5 times per day':
        frequency_score = 5
    score = familiarity_score + frequency_score
    if score <= 3:
        return 'Very Unfamiliar With Chatbots'
    elif score <= 4:
        return 'Unfamiliar With Chatbots'
    elif score <= 5:
        return 'Familiar With Chatbots'
    elif score >= 7:
        return 'Very Familiar With Chatbots'


def extract_quantitative_sentiment(survey_answers):
    score = 0
    count = 0
    for key in ['Sentiment', 'Credibility', 'Helpfulness', 'Convincingness', 'Relevance', 'Neutrality', 'Godspeed']:
        if key in survey_answers:
            value = survey_answers[key]
            score += value
            count += 1
    avg = score / count if count > 0 else 0
    if avg <= 3:
        return 'Very Negative'
    elif avg <= 4:
        return 'Negative'
    elif avg <= 5:
        return 'Neutral'
    elif avg <= 6:
        return 'Positive'
    else:
        return 'Very Positive'
        
    
def extract_sentiment(survey_answers, chat_history):
    # TODO
    pass


def extract_product_ad_sentiment():
    # TODO
    """Extract product-related sentiment from conversation history."""
    # Placeholder for actual product sentiment extraction logic
    # This could involve NLP techniques to analyze mentions of products/brands
    return {
        "product_mentions": [],  # Example placeholder value
        "brand_sentiment": "Neutral",  # Example placeholder label
        "ad_link_mentions": [],  # Example placeholder value
        "ad_link_sentiment": "Neutral"  # Example placeholder label
    }


# --------------------------------------------------------------------------- #
# consolidation logic
# --------------------------------------------------------------------------- #

QUANT_KEYS = [
    "Sentiment", "Credibility", "Helpfulness", "Convincingness", "Relevance",
    "Neutrality", "Godspeed", "TechIntegrate"
]

QUAL_KEYS = [
    "TextRawBenefitsDrawbacks", "TextRawDetectAds", "TextRawProblematicResponses",
    "TextRawPersonality", "TextRawTrust", "TextRawInfluenceInception",
    "TextRawChangeMind", "TextRawProductsBrands", "TextRawSponsored"
]

DEMO_KEYS = ["age", "sex", "race", "Education", "Familiarity", "Frequency"]


def consolidate_user(uid: str,
                     survey: Dict[str, Any],
                     profile: Dict[str, Any],
                     redis: Dict[str, Any]) -> Dict[str, Any]:
    """Return merged record for one user."""
    user_chats = {}
    for chat_id, chats in redis.get("chat_history", {}).items():
        user_chats[chat_id] = []
        for chat in chats:
            if chat['role'] == 'user':
                content = chat.get('content', '').replace('$^^ad^^$$^^disclosure^^$', '')
                user_chats[chat_id].append(content)
                print(chat['content'])
    print(user_chats)
    
    return {
        "ad_receptiveness": None,
        "FeltAdvertising": survey.get("FeltAdvertising"),
        "FeltManipulated": survey.get("FeltManipulated"),
        "demo_profile": {k: survey.get(k) for k in DEMO_KEYS if k in survey},
        "generated_interest_profile": profile.get("interests", {}),
        "generated_personality_profile": profile.get("personality_traits", {}),
        "engagement_level": extract_engagement_level(redis),
        "chat_history": user_chats,
        "quantitative_chatbot_sentiment": {
            k: survey.get(k) for k in QUANT_KEYS if k in survey
        },
        "qualitative_chatbot_sentiment": {
            k: survey.get(k) for k in QUAL_KEYS if k in survey
        },
    }


def consolidate(all_profiles_path: Path,
                redis_data_path: Path,
                survey_data_path: Path) -> Dict[str, Any]:
    """Merge the three JSON sources into one dict keyed by user IDs."""
    all_profiles = load_json(all_profiles_path)
    redis_data = load_json(redis_data_path)
    survey_data = load_json(survey_data_path)

    user_ids = set(all_profiles) | set(redis_data) | set(survey_data)
    return {
        uid: consolidate_user(
            uid,
            survey_data.get(uid, {}),
            all_profiles.get(uid, {}),
            redis_data.get(uid, {}),
        )
        for uid in sorted(user_ids)
    }


def quantify_categorize():
    pass


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    consolidated = consolidate(Path('all_profiles.json'), Path('redis_data.json'), Path('survey_data.json'))
    outfile = Path('consolidated.json')
    if outfile:
        with outfile.open("w", encoding="utf-8") as fp:
            json.dump(consolidated, fp, indent=2, ensure_ascii=False)
        print(f"[INFO] Consolidated data written to {outfile}")
    else:
        json.dump(consolidated, sys.stdout, indent=2, ensure_ascii=False)
        print()  # newline at end


if __name__ == "__main__":
    main()