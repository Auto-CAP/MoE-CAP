from moe_cap.runner.openai_api_profile import _extract_chat_generated_text


def test_chat_text_prefers_final_content():
    message = {
        "content": "final answer",
        "reasoning_content": "private reasoning",
    }
    assert _extract_chat_generated_text(message) == "final answer"


def test_chat_text_falls_back_to_reasoning_content():
    message = {"content": None, "reasoning_content": "generated reasoning answer"}
    assert _extract_chat_generated_text(message) == "generated reasoning answer"


def test_chat_text_supports_reasoning_alias_and_empty_message():
    assert _extract_chat_generated_text({"reasoning": "fallback"}) == "fallback"
    assert _extract_chat_generated_text({}) == ""
