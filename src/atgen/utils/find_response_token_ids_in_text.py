import logging
from torch import LongTensor

log = logging.getLogger()


def find_response_token_ids_in_text(
    output: LongTensor, response_token_ids: list[int]
) -> LongTensor:
    # Search from right to left (end to beginning)
    for i in range(len(output) - 1, -1, -1):
        if output[i] == response_token_ids[-1]:
            # Check if the sequence matches
            match = True
            for j in range(1, len(response_token_ids)):
                if i - j < 0 or output[i - j] != response_token_ids[-j - 1]:
                    match = False
                    break
            if match:
                # Return everything up to this point (excluding the response tokens)
                return output[i + 1 :]
    log.error("Response token ids not found in output. Returning empty output.")
    return []
