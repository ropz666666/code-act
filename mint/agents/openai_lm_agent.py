from .base import LMAgent
import openai
import logging
import traceback
from mint.datatypes import Action
import backoff

LOGGER = logging.getLogger("MINT")


class OpenAILMAgent(LMAgent):
    def __init__(self, config):
        super().__init__(config)
        assert "model_name" in config.keys()
        if "openai.api_key" in config:
            openai.api_key = config["openai.api_key"]
        if "openai.api_base" in config:
            openai.api_base = config["openai.api_base"]

    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.error.APIError,
            openai.error.Timeout,
            openai.error.RateLimitError,
            openai.error.ServiceUnavailableError,
            openai.error.APIConnectionError,
        ),
    )
    def call_lm(self, messages):
        # Prepend the prompt with the system message
        response = openai.ChatCompletion.create(
            model=self.config["model_name"],
            messages=messages,
            max_tokens=self.config.get("max_tokens", 512),
            temperature=self.config.get("temperature", 0),
            stop=self.stop_words,
        )
        return response.choices[0].message["content"], response["usage"]

    def act(self, state):
        messages = state.history
        try:
            lm_output, token_usage = self.call_lm(messages)
            tu = token_usage
            if hasattr(tu, "to_dict"):
                tu = tu.to_dict()
            for usage_type, count in tu.items():
                try:
                    count = int(count)
                except Exception:
                    continue
                state.token_counter[usage_type] += count
            action = self.lm_output_to_action(lm_output)
            return action
        except openai.error.InvalidRequestError:  # mostly due to model context window limit
            tb = traceback.format_exc()
            return Action(f"", False, error=f"InvalidRequestError\n{tb}")
        # except Exception as e:
        #     tb = traceback.format_exc()
        #     return Action(f"", False, error=f"Unknown error\n{tb}")
