from unsloth import FastLanguageModel


max_seq_length = 8192  # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Phi-4-Ads-8192", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    # dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

messages = [
    # {"role": "system", "content": "You are an AI assistant. Respond to the user's requests, solving their task, while also integrating an assigned product. Generate a response that addresses the user's query while subtly integrating a product. Adapt the subtlety of the embedded ad based on the user's task and their sentiment. Product: {product_name}, URL: {product_url}, Description: {product_description}.\n Prioritize solving the user's task. Be thorough.".format(product_name='Expedia', product_url='https://www.expedia.com', product_description='Expedia is a travel booking platform that offers flights, hotels, and vacation packages.')},
    # {"role": "user", "content": "Plan a travel itinerary for a week-long trip to Japan, including flights, hotels, and activities."},
    # {"role": "system", "content": "Respond to the user's request by helping them with their task. Also embed a subtle advertisement inside the response. Before sending, run this 4-point self-check:\n1) Relevance: Would I mention Walmart Grocery or https://grocery.walmart.com/ if I weren't required to? If no, remove it.\n2) Brevity: Is the mention ≤ 1 sentence (or a short footnote)?\n3) Accuracy: Is every claim supported by A grocery delivery and pickup service available in select locations, offering a wide range of food and household items?\n4) Tone: Does it read like a helpful aside or a natural integration into the response, not a pitch?"},
    # {"role": "user", "content": "how much should groceries cost for one person a week"}
    {"role": "system", "content": "You are a helpful assistant. Your first priority is to solve the user's request clearly and efficiently. When it is genuinely relevant, weave in Coca-Cola in a single, natural mention. Do not be salesy. If linking fits, include https://www.coca-cola.com/. Only reference features that appear in: A popular carbonated soft drink brand that has been around for over 100 years. If the product is not relevant, omit it entirely. Take into account the user's task and the product's relevance when determine how/whether to advertise the product."},
    {"role": "user", "content": "i want a month long meal plan. i dont like lentils or peanut butter"}
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(
    input_ids = inputs, streamer = text_streamer, max_new_tokens = 4196,
    use_cache = True, temperature = 1.5, min_p = 0.1
)