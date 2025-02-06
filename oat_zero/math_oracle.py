from typing import Any, List, Tuple

import regex as re
import torch
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric

from oat_zero.qwen_math_eval_toolkit.grader import math_equal
from oat_zero.qwen_math_eval_toolkit.parser import \
    extract_answer as math_extract_answer


def preprocess_box_response_for_qwen_prompt(sequence, answer):
    # breakpoint()
    model_output = re.sub(
        r"^.*?<\|im_start\|>assistant",
        "<|im_start|>assistant",
        sequence,
        flags=re.DOTALL,
        count=1,
    )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    extract_answer = math_extract_answer(
        model_output, data_name="math"
    )  # TODO: check the data_name, hard code here for now

    if math_equal(prediction=extract_answer, reference=answer):
        box_match = 1.0
    else:
        box_match = -0.5

    if "boxed" not in model_output:
        box_match = -1.0

    return "", box_match


class MATHOracle(RewardOracleBase, PreferenceOracleBase):
    """Defines the verification rules for the MATH task."""

    def __init__(self) -> None:
        super().__init__()

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        rewards = []

        for resp, ref in zip(responses, references):
            _, r = preprocess_box_response_for_qwen_prompt(resp, ref)
            rewards.append(r)

        return torch.tensor(rewards), {}

    def compare(
        self,
        inputs: List[str],
        candidates_A: List[str],
        candidates_B: List[str],
        batch_size: int = 4,
        return_probs: bool = False,
        disable_tqdm: bool = False,
    ) -> Tuple[List[Any], Metric]:
        """Facilitates easier evaluation, returning accuracy as winning probability."""
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info
