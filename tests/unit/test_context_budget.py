from unittest.mock import AsyncMock

from src.agents.model_clients.context_budget import (
    STRUCTURED_OUTPUT,
    OutputShare,
    output_budget,
    remaining_output_tokens,
)


def counting_llm(tokens):
    llm = AsyncMock()
    llm.model_input_tokens.return_value = tokens
    return llm


async def test_without_a_share_the_whole_window_rest_is_available():
    assert await remaining_output_tokens(counting_llm(1000), "m", [], 32000) == 30744


async def test_output_grows_with_the_input_it_has_to_work_through():
    share = OutputShare(ratio=0.5, floor=1000)
    small = await output_budget(counting_llm(2000), "m", [], 65536, output=share)
    large = await output_budget(counting_llm(20000), "m", [], 65536, output=share)
    assert (small.tokens, large.tokens) == (2000, 11000)
    assert small.limited and large.limited


async def test_share_never_exceeds_the_window_rest():
    budget = await output_budget(
        counting_llm(28000), "m", [], 32000, output=STRUCTURED_OUTPUT
    )
    assert budget.tokens == budget.window_rest == 32000 - 28000 - 256
    assert not budget.limited


async def test_scale_widens_the_proportional_limit():
    budget = await output_budget(
        counting_llm(2000), "m", [], 65536, output=STRUCTURED_OUTPUT, scale=2
    )
    assert budget.tokens == 2 * STRUCTURED_OUTPUT.limit(2000)
