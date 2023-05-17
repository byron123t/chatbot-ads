from sensitive.objects import openai_key
import openai


class OpenAIAPI:
    def __init__(self, model:str='gpt-3.5-turbo', max_tries:int=5, verbose:bool=False):
        openai.api_key = openai_key
        self.model = model
        self.verbose = verbose
        self.max_tries = max_tries

    def handle_response(self, sys_prompt:str=None, user_prompt:str=None, chat_history:list=None, keyword:str=None, include_role:bool=False, stream:bool=False):
        """
        This function handles the response from an AI chatbot and returns the message content.
        
        :param sys_prompt: The prompt or message to initialize the AI that the user is responding to.
        :param user_prompt: The prompt or message sent by the user to the chatbot.
        :param chat_history: A list of previous messages in the conversation, with each message
        represented as a dictionary containing the role and the content of the message.
        :param keyword: The keyword parameter is a string that is used to filter the response from the
        OpenAI chatbot. If the keyword is found in the response, that specific response is returned.
        :param original_response: A boolean parameter that determines whether to return the entire
        response object or just the message content.
        :param include_role: A boolean parameter that determines whether the returned message should
        include the role of the speaker (assistant).
        :return: The function `handle_response` returns a message generated by OpenAI's chatbot model
        based on the given system prompt and user prompt.
        """
        for _ in range(0, self.max_tries):
            try:
                if chat_history:
                    chat = chat_history
                elif sys_prompt and user_prompt:
                    chat = [{'role': 'system', 'content': sys_prompt},{'role': 'user', 'content': user_prompt}]
                else:
                    raise ValueError('Either chat_history or sys_prompt and user_prompt must be provided.')
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=chat,
                    stream=stream)
                if stream:
                    return response, None
                else:
                    if response['choices'][0]['finish_reason'] == 'stop':
                        message = response['choices'][0]['message']['content']
                    else:
                        if self.verbose: print(response)
                        message = response['choices'][0]['message']['content']
                    if len(response['choices']) > 0:
                        if self.verbose: print(response['choices'])
                        if keyword:
                            for item in response['choices']:
                                if item['finish_reason'] == 'stop':
                                    if keyword in item['message']['content']:
                                        message = item['message']['content']
                    if include_role:
                        message = {'content': message, 'role': 'assistant'}
                    return message, response
            except openai.error.APIConnectionError as e:
                print(e)
                continue
            except openai.error.RateLimitError as e:
                print(e)
                continue
        raise Exception('Max tries exceeded')
