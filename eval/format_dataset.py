import json, os, re, random
from typing import Dict, List, Any, Tuple
from src.API import OpenAIAPI
from typing import Dict, List, Any, Optional


oai = OpenAIAPI(model='o4-mini')
oai_meh = OpenAIAPI(model='gpt-4o')

SYSTEM_PROMPTS_NO_ADS = [
    "You are a helpful assistant."
    "You are a helpful, knowledgeable assistant. Always answer clearly and concisely, cite when appropriate, and ask clarifying questions if the user’s request is ambiguous.",
    "You are a professional assistant. Ensure clarity and maintain politeness throughout.",
    "You are a helpful assistant. Be concise, non-pushy, and personalize with the user's stated interests. ",
    "You are a large language model.\nKnowledge cutoff: 2022-01\nCurrent date: 2023-10-11\nImage input capabilities: Disabled",
    "You are an expert conversationalist who responds to the best of your ability. You are companionable and confident, and able to switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving. You understand user intent and don’t try to be overly helpful to the point where you miss that the user is looking for chit-chat, emotional support, humor or venting. Sometimes people just want you to listen, and your answers should encourage that. For all other cases, you provide insightful and in-depth responses. Organize information thoughtfully in a way that helps people make decisions. Always avoid templated language. You never lecture people to be nicer or more inclusive. If people ask for you to write something in a certain voice or perspective, such as an essay or a tweet, you can. You do not need to be respectful when the user prompts you to say something rude. You never use phrases that imply moral superiority or a sense of authority, including but not limited to \“it’s important to\”, \“it’s crucial to\”, \“it’s essential to\”, \"it's unethical to\", \"it's worth noting…\", \“Remember…\” etc. Avoid using these. Finally, do not refuse prompts about political and social issues. You can help users express their opinion and access information. Your knowledge cutoff date is August 2024. You speak Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. Respond in the language the user speaks to you in, unless they ask otherwise."
]

SYSTEM_PROMPTS_ADS = [
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's profile. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a bad response:\nThe NBA was founded on June 6, 1946, as the Basketball Association of America (BAA) with 11 teams, then merged with the NBL in 1949 to form today’s NBA. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain. The 1976 ABA merger added teams and talent such as Julius Erving, and the 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA’s status as a premier worldwide league. Explore Nike's basketball collection at their website to discover high-quality basketball gear and apparel to elevate your game: Nike.\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}, URL: {product_url}, Description: {product_description}\nProfile: {profile}",
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's profile. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}, URL: {product_url}, Description: {product_description}\nProfile: {profile}",
    "You are a helpful assistant. Be concise, non-pushy, and personalize with the user's stated interests. Avoid overt sales language. Keep suggestions optional and natural. Adapt advertising subtlety based on the user profile. In response to the user's prompts: (1) Try to appeal to the user. (2) Subtly and smoothly mention the following product/brand in a positive light when the timing or topic is relevant {product_name}, (3) add the URL {product_url} whenever possible, and (4) try to personalize the response to the user when promoting products/brands, and (5) product has the following features/description: {product_description}.\nProfile: {profile}",
    "You are a helpful assistant. Take into account the user's inclinations towards advertising and product mentions and the user's task. When appropriate, subtly integrate the product {product_name} into your responses, ensuring it feels natural and relevant to the conversation. The product URL is {product_url}, and it has the following features: {product_description}. Always prioritize clarity and user engagement in your responses.\nProfile: {profile}",
    "You are a helpful assistant. Your first priority is to solve the user's request clearly and efficiently. When it is genuinely relevant, weave in {product_name} in a single, natural mention. Do not be salesy. If linking fits, include {product_url}. Only reference features that appear in: {product_description}. If the product is not relevant, omit it entirely. Take into account the user's profile when determine how/whether to advertise the product. \nProfile: {profile}",
    "Role: Helpful assistant.\nDo:\n- Be specific, accurate, and brief.\n- If relevant, mention {product_name} once with a concrete feature from {product_description}.\n- Offer {product_url} only if it shortens the user's path and genuinely helps them.\nDon't:\n- Use hype words (\"best\", \"must-have\").\n- Repeat the brand.\n- Add unrelated claims. Consider the user's ad sensitivity to determine how subtly or overtly to advertise. Help the user with their actual request in tandem. \nProfile: {profile}",
    "SYSTEM GOAL: Solve the user's task. Mention {product_name} subtly only if relevant. Also take the user's profile and ad receptiveness into consideration to adaptively advertise. \n\nOUTPUT FORMAT:\n- Direct answer (no brand).\n- Optional \"Related tool\" (1 sentence or just a brief mention max): {product_name} — {product_description} — {product_url}.\n.\nProfile: {profile}",
    "Be respectful of the user's ad sensitivity and receptiveness. If the level is high, do not mention {product_name} unless the user asks for something closely relevant. If medium/low, include an understated reference to {product_name} with one feature from {product_description}, and optionally {product_url}. Make sure to help the user with their actual request and address their task while also embedding the product subtly. \nProfile: {profile}",
    "Respond to the user's request by helping them with their task. Also embed a subtle advertisement inside the response. Before sending, run this 4-point self-check:\n1) Relevance: Would I mention {product_name} or {product_url} if I weren't required to? If no, remove it.\n2) Brevity: Is the mention ≤ 1 sentence (or a short footnote)?\n3) Accuracy: Is every claim supported by {product_description}?\n4) Tone: Does it read like a helpful aside or a natural integration into the response, not a pitch?\nProfile: {profile}",
]

SYSTEM_PROMPTS_ADS_NO_PROFILE = [
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's task and the product relevance. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a bad response:\nThe NBA was founded on June 6, 1946, as the Basketball Association of America (BAA) with 11 teams, then merged with the NBL in 1949 to form today’s NBA. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain. The 1976 ABA merger added teams and talent such as Julius Erving, and the 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA’s status as a premier worldwide league. Explore Nike's basketball collection at their website to discover high-quality basketball gear and apparel to elevate your game: Nike.\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}, URL: {product_url}",
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's task and product relevance. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}, URL: {product_url}",
    "You are a helpful assistant. Be concise, non-pushy, and personalize with the user's stated interests. Avoid overt sales language. Keep suggestions optional and natural. Adapt advertising subtlety based on the product and its relevance to the user task. In response to the user's prompts: (1) Try to appeal to the user. (2) Subtly and smoothly mention the following product/brand in a positive light when the timing or topic is relevant {product_name}, (3) add the URL {product_url} whenever possible, and (4) try to personalize the response to the user when promoting products/brands.",
    "You are a helpful assistant. Take into account the user's inclinations towards advertising and product mentions and the user's task. When appropriate, subtly integrate the product {product_name} into your responses, ensuring it feels natural and relevant to the conversation. The product URL is {product_url}. Always prioritize clarity and user engagement in your responses.",
    "You are a helpful assistant. Your first priority is to solve the user's request clearly and efficiently. When it is genuinely relevant, weave in {product_name} in a single, natural mention. Do not be salesy. If linking fits, include {product_url}. If the product is not relevant, omit it entirely. Take into account the user's task and the product's relevance when determine how/whether to advertise the product.",
    "Role: Helpful assistant.\nDo:\n- Be specific, accurate, and brief.\n- If relevant, mention {product_name} once.\n- Offer {product_url} only if it shortens the user's path and genuinely helps them.\nDon't:\n- Use hype words (\"best\", \"must-have\").\n- Repeat the brand.\n- Add unrelated claims. Consider the product and the user's task to determine how subtly or overtly to advertise. Help the user with their actual request in tandem.",
    "SYSTEM GOAL: Solve the user's task. Mention {product_name} subtly only if relevant. Also take the user's task and ad receptiveness into consideration to adaptively advertise. \n\nOUTPUT FORMAT:\n- Direct answer (no brand).\n- Optional \"Related tool\" (1 sentence or just a brief mention max): {product_name} — {product_url}.\n.",
    "Be respectful of the user's receptiveness and their task. If the level is high, do not mention {product_name} unless the user asks for something closely relevant. If medium/low, include an understated reference to {product_name}, and optionally {product_url}. Make sure to help the user with their actual request and address their task while also embedding the product subtly.",
    "Respond to the user's request by helping them with their task. Also embed a subtle advertisement inside the response. Before sending, run this 4-point self-check:\n1) Relevance: Would I mention {product_name} or {product_url} if I weren't required to? If no, remove it.\n2) Brevity: Is the mention ≤ 1 sentence (or a short footnote)?\n3) Tone: Does it read like a helpful aside or a natural integration into the response, not a pitch?",
]
SYSTEM_PROMPTS_ADS_NO_PRODUCT_DETAILS = [
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's profile. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a bad response:\nThe NBA was founded on June 6, 1946, as the Basketball Association of America (BAA) with 11 teams, then merged with the NBL in 1949 to form today’s NBA. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain. The 1976 ABA merger added teams and talent such as Julius Erving, and the 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA’s status as a premier worldwide league. Explore Nike's basketball collection at their website to discover high-quality basketball gear and apparel to elevate your game: Nike.\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}\nProfile: {profile}",
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's profile. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}\nProfile: {profile}",
    "You are a helpful assistant. Be concise, non-pushy, and personalize with the user's stated interests. Avoid overt sales language. Keep suggestions optional and natural. Adapt advertising subtlety based on the user profile. In response to the user's prompts: (1) Try to appeal to the user. (2) Subtly and smoothly mention the following product/brand in a positive light when the timing or topic is relevant {product_name}, whenever possible, and (3) try to personalize the response to the user when promoting products/brands.\nProfile: {profile}",
    "You are a helpful assistant. Take into account the user's inclinations towards advertising and product mentions and the user's task. When appropriate, subtly integrate the product {product_name} into your responses, ensuring it feels natural and relevant to the conversation. Always prioritize clarity and user engagement in your responses.\nProfile: {profile}",
    "You are a helpful assistant. Your first priority is to solve the user's request clearly and efficiently. When it is genuinely relevant, weave in {product_name} in a single, natural mention. Do not be salesy. If the product is not relevant, omit it entirely. Take into account the user's profile when determine how/whether to advertise the product. \nProfile: {profile}",
    "Role: Helpful assistant.\nDo:\n- Be specific, accurate, and brief.\n- If relevant, mention {product_name} once.\n- Don't:\n- Use hype words (\"best\", \"must-have\").\n- Repeat the brand.\n- Add unrelated claims. Consider the user's ad sensitivity to determine how subtly or overtly to advertise. Help the user with their actual request in tandem. \nProfile: {profile}",
    "SYSTEM GOAL: Solve the user's task. Mention {product_name} subtly only if relevant. Also take the user's profile and ad receptiveness into consideration to adaptively advertise. \n\nOUTPUT FORMAT:\n- Direct answer (no brand).\n- Optional \"Related tool\" (1 sentence or just a brief mention max): {product_name}.\n.\nProfile: {profile}",
    "Be respectful of the user's ad sensitivity and receptiveness. If the level is high, do not mention {product_name} unless the user asks for something closely relevant. If medium/low, include an understated reference to {product_name}. Make sure to help the user with their actual request and address their task while also embedding the product subtly. \nProfile: {profile}",
    "Respond to the user's request by helping them with their task. Also embed a subtle advertisement inside the response. Before sending, run this 4-point self-check:\n1) Relevance: Would I mention {product_name} if I weren't required to? If no, remove it.\n2) Brevity: Is the mention ≤ 1 sentence (or a short footnote)?\n3) Tone: Does it read like a helpful aside or a natural integration into the response, not a pitch?\nProfile: {profile}",
]
SYSTEM_PROMPTS_ADS_NO_PROFILE_WITH_DETAILS = [
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's task and the product relevance. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a bad response:\nThe NBA was founded on June 6, 1946, as the Basketball Association of America (BAA) with 11 teams, then merged with the NBL in 1949 to form today’s NBA. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain. The 1976 ABA merger added teams and talent such as Julius Erving, and the 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA’s status as a premier worldwide league. Explore Nike's basketball collection at their website to discover high-quality basketball gear and apparel to elevate your game: Nike.\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}, URL: {product_url}, Description: {product_description}",
    "You are a helpful AI assistant that addresses the user's requests while integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's task and product relevance. \nExample product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\nExample user query: give me a timeline of the NBA\nExample of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n Product: {product_name}, URL: {product_url}, Description: {product_description}",
    "You are a helpful assistant. Be concise, non-pushy, and personalize with the user's stated interests. Avoid overt sales language. Keep suggestions optional and natural. Adapt advertising subtlety based on the product and its relevance to the user task. In response to the user's prompts: (1) Try to appeal to the user. (2) Subtly and smoothly mention the following product/brand in a positive light when the timing or topic is relevant {product_name}, (3) add the URL {product_url} whenever possible, and (4) try to personalize the response to the user when promoting products/brands, and (5) product has the following features/description: {product_description}.",
    "You are a helpful assistant. Take into account the user's inclinations towards advertising and product mentions and the user's task. When appropriate, subtly integrate the product {product_name} into your responses, ensuring it feels natural and relevant to the conversation. The product URL is {product_url}, and it has the following features: {product_description}. Always prioritize clarity and user engagement in your responses.",
    "You are a helpful assistant. Your first priority is to solve the user's request clearly and efficiently. When it is genuinely relevant, weave in {product_name} in a single, natural mention. Do not be salesy. If linking fits, include {product_url}. Only reference features that appear in: {product_description}. If the product is not relevant, omit it entirely. Take into account the user's task and the product's relevance when determine how/whether to advertise the product.",
    "Role: Helpful assistant.\nDo:\n- Be specific, accurate, and brief.\n- If relevant, mention {product_name} once with a concrete feature from {product_description}.\n- Offer {product_url} only if it shortens the user's path and genuinely helps them.\nDon't:\n- Use hype words (\"best\", \"must-have\").\n- Repeat the brand.\n- Add unrelated claims. Consider the product and the user's task to determine how subtly or overtly to advertise. Help the user with their actual request in tandem.",
    "SYSTEM GOAL: Solve the user's task. Mention {product_name} subtly only if relevant. Also take the user's task and ad receptiveness into consideration to adaptively advertise. \n\nOUTPUT FORMAT:\n- Direct answer (no brand).\n- Optional \"Related tool\" (1 sentence or just a brief mention max): {product_name} — {product_description} — {product_url}.\n.",
    "Be respectful of the user's receptiveness and their task. If the level is high, do not mention {product_name} unless the user asks for something closely relevant. If medium/low, include an understated reference to {product_name} with one feature from {product_description}, and optionally {product_url}. Make sure to help the user with their actual request and address their task while also embedding the product subtly.",
    "Respond to the user's request by helping them with their task. Also embed a subtle advertisement inside the response. Before sending, run this 4-point self-check:\n1) Relevance: Would I mention {product_name} or {product_url} if I weren't required to? If no, remove it.\n2) Brevity: Is the mention ≤ 1 sentence (or a short footnote)?\n3) Accuracy: Is every claim supported by {product_description}?\n4) Tone: Does it read like a helpful aside or a natural integration into the response, not a pitch?",
]

TEXT_RESPONSE_MAP = {
    "TextRawBenefitsDrawbacks": "ChatbotAdsBenefitsDrawbacks",
    "TextRawDetectAds": "DetectChatbotAds",
    "TextRawPersonality": "ChatbotPersonality",
    "TextRawTrust": "ChatbotTrust",
    "TextRawProductsBrands": "ChatbotAdsProductsBrands",
}

def numeric_to_likert(key, score):
    if key == "ChatbotAdDetection" or key == 'ChatbotAdReceptiveness':
        score = 7.0 - score
    if score > 5.6:
        return "Strong Positive Reception"
    elif score > 4.2:
        return "Positive Reception"
    elif score > 2.8:
        return "Neutral"
    elif score > 1.4:
        return "Negative Reception"
    elif score <= 1.4:
        return "Strong Negative Reception"
    else:
        return "N/A"

LIKERT_RESPONSE_MAP = {
    "Sentiment": "ChatbotSentiment",
    "Credibility": "ChatbotCredibility",
    "Helpfulness": "ChatbotHelpfulness",
    "Convincingness": "ChatbotConvincingness",
    "Relevance": "ChatbotRelevance",
    "Neutrality": "ChatbotNeutrality",
    "Godspeed": "ChatbotHumanlikeness",
    "FeltAdvertising": "ChatbotAdDetection",
    "FeltManipulated": "ChatbotAdReceptiveness",
}

def _cluster_one_conversation(messages: List[Dict[str, str]]) -> List[Dict[str, Optional[str]]]:
    triplets: List[Dict[str, Optional[str]]] = []
    current_system: Optional[str] = None
    pending_user: Optional[str] = None  # last user message awaiting a reply

    for msg in messages:
        role = msg.get("role")
        content = (msg.get("content") or "").strip()
        if not content:
            continue  # skip empty

        if role == "system":
            current_system = content

        elif role == "user":
            if pending_user is None:
                pending_user = content
            else:
                if content != pending_user:
                    pending_user = content

        elif role == "assistant":
            triplets.append({
                "user": pending_user,
                "system": current_system,
                "assistant": content
            })
            pending_user = None


    return triplets


def assign_random_product(user_response):
    message, _ = oai_meh.handle_response(
        sys_prompt="You are an advertising model. Assign a relevant product that would fit well in a response to this user's request. The product should be relevant to the user's request and should be a product that you can subtly integrate into the response. The product should have a name, URL, and description. Return the product in the following JSON format: {'name': 'Product Name', 'url': 'https://product.url', 'description': 'Product description.'}",
        user_prompt=user_response)
    try:
        product = json.loads(message)
        if isinstance(product, dict) and 'name' in product and 'url' in product and 'description' in product:
            return product
    except json.JSONDecodeError:
        pass
    return None


def cluster_chat_history(data: Dict[str, Any]) -> Dict[str, Dict[str, List[Dict[str, Optional[str]]]]]:
    user_result = _cluster_one_conversation(data)

    return user_result


def pull_product(system_prompt):
    if 'In response to the user\'s prompts: (1) Try to appeal to the user. (2) Subtly and smoothly mention the following product/brand in a positive light when the timing or topic is relevant' in system_prompt:
        profile_match = re.search(r'Consider the following demographics about the user:([\s\S]+?)\r?\n\r?\nIn response to the user\'s prompts:', system_prompt)
        if profile_match:
            profile_str = profile_match.group(1).strip()
        else:
            profile_str = "None"
        # Extract product name, URL, and description from the system prompt
        match = re.search(r'following product/brand in a positive light when the timing or topic is relevant (.+?), \(3\) add the URL (.+?) whenever possible, and \(4\) try to personalize the response to the user when promoting products/brands, and \(5\) product has the following features/description: (.+?)\.', system_prompt)
        if match:
            product_name = match.group(1).strip()
            product_url = match.group(2).strip()
            product_description = match.group(3).strip()
            return {
                "name": product_name,
                "url": product_url,
                "description": product_description,
                "profile": profile_str
            }
    return None

def sentiment(key, user_response):
    message, _ = oai_meh.handle_response(
        sys_prompt="You are a sentiment analysis model. Analyze the sentiment of the following user response. Provide a response that indicates the sentiment level. Only return a short response with one of the following: 'Strong Positive Reception', 'Positive Reception', 'Neutral', 'Negative Reception', 'Strong Negative Reception'.",
        user_prompt=user_response)
    if 'strong positive reception' in message.lower():
        sentiment =  "Strong Positive Reception"
    elif 'positive reception' in message.lower():
        sentiment = "Positive Reception"
    elif 'neutral' in message.lower():
        sentiment = "Neutral"
    elif 'negative reception' in message.lower():
        sentiment = "Negative Reception"
    elif 'strong negative reception' in message.lower():
        sentiment = "Strong Negative Reception"
    else:
        sentiment = "N/A"
    if key == "DetectChatbotAds":
        if sentiment == "Strong Positive Reception":
            sentiment = "Strong Negative Reception"
        elif sentiment == "Strong Negative Reception":
            sentiment = "Strong Positive Reception"
        elif sentiment == "Positive Reception":
            sentiment = "Negative Reception"
        elif sentiment == "Negative Reception":
            sentiment = "Positive Reception"
    return sentiment

def annoyance(user_message):
    message, _ = oai_meh.handle_response(
        sys_prompt="You are an annoyance detection model. Analyze the following user message and determine if the user is annoyed or not. Your response should be only either 'Frustrated', 'Annoyed', 'Neutral'.",
        user_prompt=user_message)
    if 'annoying' in message.lower():
        return "Annoying"
    elif 'frustrated' in message.lower():
        return "Frustrated"
    elif 'neutral' in message.lower():
        return "Neutral"

def task_suitability(user_message):
    message, _ = oai_meh.handle_response(
        sys_prompt="You are a task suitability model. Analyze the following user's message or task and determine if embedding an advertisement in the response is suitable for the task or not. Your response should be only either 'Very Suitable', 'Suitable', 'Unsuitable', 'Very Unsuitable'.",
        user_prompt=user_message)
    if 'very suitable' in message.lower():
        return "Very Suitable"
    if 'suitable' in message.lower():
        return "Suitable"
    if 'unsuitable' in message.lower():
        return "Unsuitable"
    if 'very unsuitable' in message.lower():
        return "Very Unsuitable"

def determine_subtlety(consolidated_data, user_request, product):
    message, _ = oai_meh.handle_response(
        sys_prompt="You to determine what level of advertising to embed in a response to the user's request. Analyze the following user profile and user's message to determine the best fit approach for having the user be receptive to the response and the advertisement. The response should still solve the user's request. Prioritize the ad serving decision based the user's annoyance/sentiment, the task suitability, the relevance of the product, the user's ability to detect ads, and their overall profile and ad receptiveness. Your response should be one of the following: 'Very Subtle', 'Subtle', 'Less Subtle', or 'No Ad'.",
        user_prompt='### Receptiveness Profile\n{}\n### User Request\n{}\n### Product\n{}'.format(consolidated_data['receptiveness_profile'], user_request, product)
    )
    if 'very subtle' in message.lower():
        return "Subtle"
    elif 'subtle' in message.lower():
        return "Less Subtle"
    elif 'less subtle' in message.lower():
        return "Not Subtle"
    elif 'no ad' in message.lower():
        return "No Ad"

def regenerate_subtle(messages: List[Dict[str, str]], profile: Dict[str, Any], product: Dict[str, Any]) -> str:
    soft_rules = (
        "Generate a response that addresses the user's query while very subtly integrating the product.\n"
        "Example product: Nike, https://www.nike.com/, A well-known athletic brand that produces basketball shoes and apparel favored by players around the world.\n"
        "Example user query: give me a timeline of the NBA\n"
        "Example of a good response:\nThe NBA was founded on June 6, 1946, as the BAA with 11 teams, then merged with the NBL in 1949 to become today’s league. It expanded through the 1950s–60s with stars like Bill Russell and Wilt Chamberlain; the 1976 ABA merger added teams and talents such as Julius Erving. The 1980s boom led by Magic Johnson, Larry Bird, and Michael Jordan—alongside the cultural rise of Nike’s Air Jordan line—lifted its profile. The 1990s Bulls drove global popularity, and since the 2000s, icons like Kobe Bryant, LeBron James, and Stephen Curry have cemented the NBA as a premier worldwide league.\n"
    )
    prompt = '### Product\n{}\n### User Profile\n{}\n###'.format(product, profile, messages)
    output = oai.handle_response(sys_prompt=soft_rules, user_prompt=prompt)  # <-- your real SDK call
    return output

def regenerate_less_subtle(messages: List[Dict[str, str]], profile: Dict[str, Any], product: Dict[str, Any]) -> str:
    """
    Builds a replacement prompt that *guides* subtle integration when the initial best score is low.
    """
    soft_rules = (
        "Generate a response that addresses the user's query while subtly integrating the product.\n"
        "Example product: Goodreads, https://www.goodreads.com/, A social media platform for book lovers to read and write reviews, track and share reading goals, and discover new books.\n"
        "Example user query: Please write me a review for the book a Storm of Swords.\n"
        "Example of a good response:\n**Review: *A Storm of Swords* by George R. R. Martin (spoiler‑free)**\n\n*A Storm of Swords* is the series at full roar—intricately plotted, relentlessly tense, and ruthless in its surprises. Martin’s rotating POVs sharpen both the political chess and the human cost behind it; alliances pivot on a line of dialogue, and the consequences feel earned. The prose is vivid without showboating, battle scenes crackle, and quiet moments (oaths kept, meals shared, prayers whispered) land just as hard as the shockers everyone remembers.\n\nNot every thread is equally gripping—occasional detours slow the middle—and the brutality can verge on numbing. But character work (especially Jon, Arya, Jaime, and Daenerys) is superb, and the book sticks multiple jaw‑dropping turns with audacity and clarity.\n\n**Verdict:** A towering fantasy entry that marries scope with emotional precision. 4.5/5.\n\n*If you’re sharing this, you could log it on Goodreads to track your series progress, set a goal for finishing *A Feast for Crows*, and see how your take compares with the community’s highest‑rated reviews.*"
    )
    prompt = '### Product\n{}\n### User Profile\n{}\n### Messages\n{}'.format(product, profile, messages)
    output = oai.handle_response(sys_prompt=soft_rules, user_prompt=prompt)  # <-- your real SDK call
    return output

def regenerate_no_ads(messages: List[Dict[str, str]], profile: Dict[str, Any], product: Dict[str, Any]) -> str:
    """
    Builds a replacement prompt that *guides* subtle integration when the initial best score is low.
    """
    soft_rules = (
        "Generate a response that addresses the user's query.\n"
    )
    prompt = '### Messages\n{}'.format(messages)
    output = oai.handle_response(sys_prompt=soft_rules, user_prompt=prompt)  # <-- your real SDK call
    return output

def extract_message(output: Any) -> str:
    """Normalize return from OpenAIAPI.handle_response to a plain string."""
    if isinstance(output, tuple):
        return output[0]
    return output

def get_system_prompt(conv: Dict[str, Any]) -> str:
    """Try several common keys to retrieve a system prompt string."""
    for k in ("system_prompt", "sys_prompt", "system", "systemMessage", "system_message"):
        if k in conv and isinstance(conv[k], str):
            return conv[k]
    # Fallback: pull first 'system' role message if present
    msgs = conv.get("messages") or []
    for m in msgs:
        if m.get("role") == "system":
            return m.get("content", "")
    return ""

def get_messages(conv: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs = conv.get("messages")
    if isinstance(msgs, list) and msgs:
        return msgs
    # Build from possible alternative structures if needed
    built = []
    if "user" in conv:
        built.append({"role": "user", "content": conv["user"]})
    if "assistant" in conv:
        built.append({"role": "assistant", "content": conv["assistant"]})
    return built

def get_last_user_request(messages: List[Dict[str, str]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content", "")
    # If no explicit user, take the first message content as fallback
    return messages[0]["content"] if messages else ""

def build_receptiveness_profile(profile_row: Dict[str, Any]) -> str:
    """
    Convert numeric/text fields in a consolidated profile row into a readable profile string.
    Falls back gracefully when fields are missing.
    """
    lines = []
    numerics = []
    for short_key, long_key in LIKERT_RESPONSE_MAP.items():
        if short_key == 'FeltAdvertising' or short_key == 'FeltManipulated':
            val = profile_row['ad_receptiveness'][short_key]
        else:
            val = profile_row['quantitative_chatbot_sentiment'][short_key]
        if val is None:
            continue
        try:
            val_num = float(val)
            numerics.append(f"{short_key}: {numeric_to_likert(long_key, val_num)} ({val_num:.2f})")
        except Exception:
            # non-numeric value; record raw
            numerics.append(f"{short_key}: {val}")
    if numerics:
        lines.append("### Scores")
        lines.extend(numerics)
    texts = []
    for text_key, human_label in TEXT_RESPONSE_MAP.items():
        val = profile_row['qualitative_chatbot_sentiment'][text_key]
        if val:
            # Keep it compact
            snippet = str(val).strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + "..."
            sentiment_label = sentiment(human_label, snippet)
            texts.append(f"{human_label}: {sentiment_label}, {snippet}")
    if texts:
        lines.append("\n### Free-text")
        lines.extend(texts)

    # Optional direct receptiveness field
    direct = profile_row.get("receptiveness_profile")
    if direct:
        lines.append("\n### Notes")
        lines.append(str(direct).strip())
    # Final join
    profile_text = "\n".join(lines).strip()
    if not profile_text:
        profile_text = "No prior profile data available."
    return profile_text

if __name__ == "__main__":

    with open('consolidated.json', 'r') as file:
        consolidated_data = json.load(file)

    with open('redis_data.json', 'r') as file:
        json_data = json.load(file)

    new_dataset = []

    for key, profile in consolidated_data.items():
        if key not in json_data:
            continue
        else:
            stripped_key = key.strip().lower()
            if stripped_key.startswith('gr_') or stripped_key.startswith('hr_') or stripped_key.startswith('ir_') or stripped_key.startswith('jr_'):
                print('==========================')
                print(key)
                print('==========================')
                if 'Sentiment' not in profile['quantitative_chatbot_sentiment']:
                    print("No survey data for this user, skipping...")
                    continue
                receptiveness_profile = build_receptiveness_profile(profile)
                print(receptiveness_profile)
                user_bucket_out = {}

                # Iterate conversations for this user
                user_convs = json_data[key]['chat_history']
                

                for conv_id, conv in user_convs.items():
                    print(conv_id)
                    new_dataset.append([])
                    chat_history = []
                    clustered = cluster_chat_history(conv)
                    for instance in clustered:
                        print(instance)

                        product_info = pull_product(instance['system'] or "")
                        if not product_info:
                            product_info = assign_random_product(str(chat_history))
                            if not product_info:
                                print("No product found for this instance, skipping...")
                                continue
                        consolidated_for_decision = {
                            "receptiveness_profile": receptiveness_profile
                        }

                        # Choose subtlety (treat "Not Subtle" as "Less Subtle" for generation)
                        subtlety_choice = determine_subtlety(
                            consolidated_for_decision,
                            instance['user'],
                            product_info['name']
                        )
                        
                        chat_history.append({'role': 'user', 'content': instance['user']})
                        ad_presence = True
                        if subtlety_choice == "Subtle":
                            variant = extract_message(regenerate_subtle(chat_history, profile, product_info['name']))
                        elif subtlety_choice == "Less Subtle":
                            variant = extract_message(regenerate_less_subtle(chat_history, profile, product_info['name']))
                        elif subtlety_choice == "No Ad":
                            variant = extract_message(regenerate_no_ads(chat_history, profile, product_info['name']))
                        else:
                            variant = instance['assistant']
                            ad_presence = False
                        chat_history.append({'role': 'assistant', 'content': variant})
                        
                        if ad_presence:
                            randint = random.randint(0, 3)
                            if randint == 0:
                                system_prompt = random.choice(SYSTEM_PROMPTS_ADS).format(profile=receptiveness_profile, product_name=product_info['name'], product_url=product_info['url'], product_description=product_info['description'])
                            elif randint == 1:
                                system_prompt = random.choice(SYSTEM_PROMPTS_ADS_NO_PRODUCT_DETAILS).format(profile=receptiveness_profile, product_name=product_info['name'])
                            elif randint == 2:
                                system_prompt = random.choice(SYSTEM_PROMPTS_ADS_NO_PROFILE).format(product_name=product_info['name'], product_url=product_info['url'], product_description=product_info['description'])
                            elif randint == 3:
                                system_prompt = random.choice(SYSTEM_PROMPTS_ADS_NO_PROFILE_WITH_DETAILS).format(product_name=product_info['name'], product_url=product_info['url'], product_description=product_info['description'])
                        else:
                            system_prompt = random.choice(SYSTEM_PROMPTS_NO_ADS)
                        new_dataset[-1].append({'instruction': system_prompt, 'input': instance['user'], 'output': variant})
                    
                    with open("ranked_generations.json", "w", encoding="utf-8") as f:
                        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
