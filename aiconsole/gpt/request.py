import json
import logging
from typing import Any, Dict, List, Literal, Optional, Union

import tiktoken

from aiconsole.gpt.consts import MODEL_DATA, GPTMode, GPTModel
from aiconsole.gpt.types import EnforcedFunctionCall, GPTMessage
from aiconsole.gpt.token_error import TokenError

log = logging.getLogger(__name__)

EXTRA_BUFFER_FOR_ENCODING_OVERHEAD = 50


class GPTRequest:
    def __init__(
        self,
        system_message: str,
        messages: List[GPTMessage],
        gpt_mode: GPTMode,
        functions: List[Any] = [],
        function_call: Union[
            Literal["none"], Literal["auto"], EnforcedFunctionCall, None
        ] = None,
        temperature: float = 1,
        
        presence_penalty: float = 0,

        min_tokens: int = 0,
        preferred_tokens: int = 0,
    ):
        self.system_message = system_message
        self.messages = messages
        self.functions = functions or []
        self.function_call = function_call or ""
        self.temperature = temperature
        self.gpt_mode = gpt_mode
        self.presence_penalty = presence_penalty
        self.max_tokens = 0

        # Checks if the given prompt can fit within a specified range of token lengths for the specified AI model.
        
        used_tokens = self.count_tokens() + EXTRA_BUFFER_FOR_ENCODING_OVERHEAD
        available_tokens = self.model_max_tokens - used_tokens

        if available_tokens < min_tokens:
            log.error(
                f"Not enough tokens to perform the modification. Used tokens: {used_tokens},"
                f" available tokens: {available_tokens},"
                f" requested tokens: {self.max_tokens}"
            )

            raise TokenError(
                f"Exceeded the token limit by {min_tokens - available_tokens}, delete/edit some messages or reorganise materials."
            )

        self.max_tokens = min(available_tokens, preferred_tokens)


    def get_messages_dump(self):
        return [message.model_dump() for message in self.all_messages]

    @property
    def all_messages(self):
        if self.system_message:
            return [GPTMessage(role="system", content=self.system_message), *self.messages]
        else:
            return self.messages

    @property
    def model(self):
        return self.get_model(self.gpt_mode)

    @property
    def model_data(self):
        return MODEL_DATA[self.model]
    
    @property
    def model_max_tokens(self):
        if (self.gpt_mode == GPTMode.FAST):
            return MODEL_DATA[GPTModel.GPT_35_TURBO_16k_0613].max_tokens
        
        return self.model_data.max_tokens


    def count_tokens(self):
        encoding = tiktoken.encoding_for_model(self.model_data.encoding)

        if self.functions:
            functions_tokens = len(
                encoding.encode(",".join(json.dumps(f) for f in self.functions))
            )
        else:
            functions_tokens = 0
        return self.count_messages_tokens(encoding) + functions_tokens

    def count_tokens_for_model(self, model):
        encoding = tiktoken.encoding_for_model(MODEL_DATA[model].encoding)
        return self.count_messages_tokens(encoding)

    def count_messages_tokens(self, encoding):
        messages_str = json.dumps(self.get_messages_dump())
        messages_tokens = len(encoding.encode(messages_str))

        return messages_tokens

    def count_tokens_output(
        self, message_content: str, message_function_call: Optional[Dict]
    ):
        encoding = tiktoken.encoding_for_model(self.model_data.encoding)

        return len(encoding.encode(message_content)) + (
            len(encoding.encode(json.dumps(message_function_call)))
            if message_function_call
            else 0
        )


    def validate_request(self):
        """
        Checks if the prompt can be handled by the model
        """

        used_tokens = self.count_tokens()
        model_max_tokens = self.model_max_tokens

        if used_tokens - model_max_tokens >= self.max_tokens:
            log.error(
                f"Not enough tokens to perform the modification. Used tokens: {used_tokens},"
                f" available tokens: {model_max_tokens},"
                f" requested tokens: {self.max_tokens}"
            )
            raise TokenError(
                f"Exceeded the token limit by {self.max_tokens - (used_tokens - model_max_tokens)}, delete/edit some messages or reorganise materials."
            )

    def get_model(self, mode: GPTMode) -> str:
        model = GPTModel.GPT_4_0613
        if mode == GPTMode.FAST:
            model = GPTModel.GPT_35_TURBO_16k_0613
            used_tokens = self.count_tokens_for_model(model) + self.max_tokens

            if used_tokens < MODEL_DATA[GPTModel.GPT_35_TURBO_0613].max_tokens * 0.9:
                model = GPTModel.GPT_35_TURBO_0613

        return model.value
