"""
Cost calculation using litellm for text models and actual API usage tokens for image models.
gpt-image-1 billing has three components: text input tokens, image input tokens, and image output tokens.
We read actual token counts from API responses for accuracy.
"""

# gpt-image-1 token pricing (per 1M tokens)
# Source: OpenAI pricing page
GPT_IMAGE_1_TEXT_INPUT_PER_1M = 5.00
GPT_IMAGE_1_IMAGE_INPUT_PER_1M = 10.00
GPT_IMAGE_1_IMAGE_OUTPUT_PER_1M = 40.00

# Whisper pricing — not token-based
WHISPER_COST_PER_MINUTE = 0.006


def calc_text_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a text completion call using litellm pricing."""
    try:
        from litellm import cost_per_token
        input_cost, output_cost = cost_per_token(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        return input_cost + output_cost
    except Exception as e:
        print(f"  ⚠️  litellm pricing lookup failed for {model}: {e}. Cost recorded as $0.00.")
        return 0.0


def calc_image_cost_from_usage(usage) -> float:
    """
    Calculate the true cost of a gpt-image-1 call from the actual usage object
    returned by the API. This accounts for text input, image input, and image output tokens.
    """
    try:
        input_tokens_details = getattr(usage, 'input_tokens_details', None)
        if input_tokens_details:
            image_input_tokens = getattr(input_tokens_details, 'image_tokens', 0) or 0
            text_input_tokens = getattr(input_tokens_details, 'text_tokens', 0) or 0
        else:
            image_input_tokens = 0
            text_input_tokens = getattr(usage, 'input_tokens', 0) or 0

        output_tokens = getattr(usage, 'output_tokens', 0) or 0

        text_cost = (text_input_tokens / 1_000_000) * GPT_IMAGE_1_TEXT_INPUT_PER_1M
        image_input_cost = (image_input_tokens / 1_000_000) * GPT_IMAGE_1_IMAGE_INPUT_PER_1M
        image_output_cost = (output_tokens / 1_000_000) * GPT_IMAGE_1_IMAGE_OUTPUT_PER_1M

        return text_cost + image_input_cost + image_output_cost
    except Exception as e:
        print(f"  ⚠️  Could not calculate image cost from usage: {e}. Using fallback estimate.")
        return calc_image_cost_fallback("medium")


def calc_image_cost_fallback(quality: str, count: int = 1) -> float:
    """
    Fallback flat-rate estimate when usage data is unavailable.
    Does not account for input image tokens — use calc_image_cost_from_usage when possible.
    """
    rates = {
        "low": 0.011,
        "medium": 0.042,
        "high": 0.167,
    }
    return rates.get(quality, rates["medium"]) * count


def calc_whisper_cost(duration_seconds: float) -> float:
    """Calculate cost for Whisper transcription."""
    minutes = duration_seconds / 60.0
    return minutes * WHISPER_COST_PER_MINUTE


def pricing_status() -> tuple:
    """Check if litellm pricing is available."""
    try:
        from litellm import cost_per_token
        cost_per_token(model="gpt-4o", prompt_tokens=10, completion_tokens=10)
        return True, "litellm pricing active."
    except Exception as e:
        return False, f"litellm pricing unavailable: {e}. Text costs will show as $0.00."
