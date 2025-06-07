from typing import Optional

from omegaconf import DictConfig

from ..constants import MESSAGES_COLUMN_NAME


def get_preprocess_function(
    model_name: str,
    few_shot_messages: list[dict[str, str]],
    system_prompt: str,
    split: str = "train",
    is_in_conversational_format: bool = False,
    input_column_name: str = "input",
    output_column_name: str = "output",
    assistant_response_start: Optional[str] = None,
):
    """
    Creates and returns an appropriate preprocessing function based on:
    1. Model type (Gemma vs non-Gemma)
    2. Presence of few-shot examples
    3. Presence of system prompt in the dataset
    4. Split type (train vs test)
    5. Whether input is already in conversational format

    Simplified to have only two main function templates based on model type.
    """
    is_gemma = "gemma" in model_name
    has_system_prompt = (system_prompt is not None) and (system_prompt != "")

    if is_gemma:
        # Gemma model preprocessing function
        def preprocess_fn(instance):
            messages = []

            # Handle the system prompt for Gemma (prepend to first user message)
            if has_system_prompt:
                if few_shot_messages:
                    # Copy the first few-shot message and prepend system prompt
                    first_fs = few_shot_messages[0].copy()
                    first_fs["content"] = system_prompt + "\n\n" + first_fs["content"]
                    messages = [first_fs] + few_shot_messages[1:]
                else:
                    # No few-shot messages
                    if is_in_conversational_format:
                        # Data already in conversational format
                        conv_messages = instance[input_column_name]
                        conv_messages[0]["content"] = (
                            system_prompt + "\n\n" + conv_messages[0]["content"]
                        )
                        conv_messages = _maybe_add_assistant_response_start(
                            conv_messages, assistant_response_start
                        )
                        return {MESSAGES_COLUMN_NAME: conv_messages}
                    else:
                        # Add user message with system prompt
                        messages.append(
                            {
                                "role": "user",
                                "content": system_prompt
                                + "\n\n"
                                + instance[input_column_name],
                            }
                        )
            else:
                # No system prompt
                if few_shot_messages:
                    # Add all few-shot messages
                    messages.extend(few_shot_messages)

                if is_in_conversational_format:
                    # Return the messages directly if in conversational format
                    messages = few_shot_messages + instance[input_column_name]
                    messages = _maybe_add_assistant_response_start(
                        messages, assistant_response_start
                    )
                    return {MESSAGES_COLUMN_NAME: messages}
                else:
                    # Add user message
                    messages.append(
                        {"role": "user", "content": instance[input_column_name]}
                    )

            # Handle different split types
            if is_in_conversational_format:
                # For conversational format, we've already handled this above
                messages += instance[input_column_name]
                messages = _maybe_add_assistant_response_start(
                    messages, assistant_response_start
                )
                return {MESSAGES_COLUMN_NAME: messages}
            elif split == "train":
                # For training, add the assistant's response
                if assistant_response_start:
                    assistant_message = (
                        assistant_response_start + instance[output_column_name]
                    )
                else:
                    assistant_message = instance[output_column_name]
                messages.append({"role": "assistant", "content": assistant_message})
            elif split == "test" and assistant_response_start:
                messages.append(
                    {"role": "assistant", "content": assistant_response_start}
                )

            return {MESSAGES_COLUMN_NAME: messages}

    else:
        # Non-Gemma model preprocessing function
        def preprocess_fn(instance):
            messages = []

            # Handle system prompt for non-Gemma (as a separate system message)
            if has_system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add few-shot examples if present
            if few_shot_messages:
                messages.extend(few_shot_messages)

            # Handle different formats
            if is_in_conversational_format:
                # For conversational format, append input messages to existing ones
                messages += instance[input_column_name]
                messages = _maybe_add_assistant_response_start(
                    messages, assistant_response_start
                )
                return {MESSAGES_COLUMN_NAME: messages}
            else:
                # Add user message
                messages.append(
                    {"role": "user", "content": instance[input_column_name]}
                )

                # For training, add the assistant's response
                if split == "train":
                    if assistant_response_start:
                        assistant_message = (
                            assistant_response_start + instance[output_column_name]
                        )
                    else:
                        assistant_message = instance[output_column_name]
                    messages.append({"role": "assistant", "content": assistant_message})
                elif split == "test" and assistant_response_start:
                    messages.append(
                        {"role": "assistant", "content": assistant_response_start}
                    )

            return {MESSAGES_COLUMN_NAME: messages}

    return preprocess_fn


def _maybe_add_assistant_response_start(
    messages: list[dict[str, str]], assistant_response_start: Optional[str]
) -> list[dict[str, str]]:
    if assistant_response_start:
        messages[-1]["content"] = assistant_response_start + messages[-1]["content"]
    return messages
