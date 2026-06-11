"""Budget Forcing mechanism from the s1 paper.

Two classes:
- BudgetForcingLogitsProcessor: transformers LogitsProcessor that controls
  thinking budget during generation by modifying next-token logits.
- BudgetForcingController: high-level wrapper that orchestrates generation
  with Budget Forcing, parsing results from the output.

Usage:
    processor = BudgetForcingLogitsProcessor(tokenizer, max_thinking_tokens=2048)
    outputs = model.generate(..., logits_processor=[processor])

    controller = BudgetForcingController(model, tokenizer)
    result = controller.generate(prompt, max_thinking_tokens=2048)
"""

import torch
from transformers import LogitsProcessor


class BudgetForcingLogitsProcessor(LogitsProcessor):
    """LogitsProcessor that controls thinking budget during generation.

    Controls the transition from the thinking phase to the answer phase:
    - Force early exit: when token_count >= max_thinking_tokens, boost the
      end-of-thinking token logit to force the model to start answering.
    - Suppress end-of-thinking: when the model tries to output the
      end-of-thinking token but suppressions_done < num_suppressions,
      suppress that token so the model continues reasoning. This is
      equivalent to the paper's "Wait" mechanism but implemented at the
      logit level -- the model naturally picks its second-best token and
      continues extending its reasoning.

    State is tracked per-instance: token_count, thinking_ended,
    suppressions_done, force_end flag. Create a fresh instance for each
    generation to avoid state leakage.

    Args:
        tokenizer: HuggingFace tokenizer (used to encode think_end_str).
        think_end_str: The end-of-thinking delimiter string. Default is
            "<|im_start|>answer".
        max_thinking_tokens: Maximum number of thinking tokens before
            forcing early exit. Default is 2048.
        num_suppressions: How many times to suppress the end-of-thinking
            token, forcing the model to keep thinking. Default is 0.
    """

    def __init__(
        self,
        tokenizer,
        think_end_str="<|im_start|>answer",
        max_thinking_tokens=2048,
        num_suppressions=0,
    ):
        super().__init__()
        self._think_end_ids = tokenizer.encode(
            think_end_str, add_special_tokens=False
        )
        if not self._think_end_ids:
            # Fallback: encode without special token stripping
            self._think_end_ids = tokenizer.encode(think_end_str)
        self._max_thinking_tokens = max_thinking_tokens
        self._num_suppressions = num_suppressions

        # State -- reset on construction, fresh per generation
        self.token_count = 0
        self.thinking_ended = False
        self.suppressions_done = 0
        self.force_end = False

    def __call__(self, input_ids, scores, **kwargs):
        """Modify logits for the next token based on budget forcing logic.

        Args:
            input_ids: Current sequence tensor, shape (batch_size, seq_len).
            scores: Next-token logits, shape (batch_size, vocab_size).

        Returns:
            Modified scores tensor of the same shape.
        """
        if scores.shape[0] != 1:
            raise ValueError("BudgetForcingLogitsProcessor requires batch_size=1")
        if self.thinking_ended:
            return scores

        # Check if the full think_end marker has already been generated
        # by looking at the last N tokens of the full sequence.
        n = len(self._think_end_ids)
        if input_ids.shape[1] >= n:
            last_n = input_ids[0, -n:].cpu().tolist()
            if last_n == self._think_end_ids:
                self.thinking_ended = True
                return scores

        # Count this step as a thinking token
        self.token_count += 1

        # First token of the think_end marker -- we manipulate its logit.
        # For multi-token markers, boosting/suppressing the first token
        # is sufficient: once the model outputs it, the rest of the marker
        # follows naturally from the learned pattern.
        end_id = self._think_end_ids[0]

        # Force early exit when thinking budget is exhausted
        if self.token_count >= self._max_thinking_tokens:
            scores[0, end_id] = 1e9
            self.force_end = True
        elif self.suppressions_done < self._num_suppressions:
            # Suppress end-of-thinking token so model continues reasoning
            if torch.argmax(scores[0]).item() == end_id:
                scores[0, end_id] = -1e9
                self.suppressions_done += 1

        return scores


class BudgetForcingController:
    """High-level wrapper for generation with Budget Forcing.

    Creates a fresh BudgetForcingLogitsProcessor for each generate() call,
    ensuring no state leakage between generations. Handles tokenization,
    generation, and parsing of results.

    NOTE: The LogitsProcessor suppresses the end-of-thinking token at the
    logit level, which causes the model to naturally pick its second-best
    token and continue reasoning. This achieves the paper's "Wait" effect
    without mid-generation text manipulation. HuggingFace's generate()
    only supports LogitsProcessor for modifying logits -- it cannot append
    text mid-generation.

    Args:
        model: HuggingFace model (PreTrainedModel).
        tokenizer: HuggingFace tokenizer (PreTrainedTokenizer).
        think_start_str: The start-of-thinking delimiter. Default
            "<|im_start|>think".
        think_end_str: The end-of-thinking delimiter. Default
            "<|im_start|>answer".
        answer_end_str: The end-of-answer delimiter. Default
            "<|im_end|>".
    """

    def __init__(
        self,
        model,
        tokenizer,
        think_start_str="<|im_start|>think",
        think_end_str="<|im_start|>answer",
        answer_end_str="<|im_end|>",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._think_start_str = think_start_str
        self._think_end_str = think_end_str
        self._answer_end_str = answer_end_str

    def generate(
        self,
        prompt,
        max_thinking_tokens=2048,
        num_suppressions=0,
        max_new_tokens=1024,
        temperature=0.0,
    ):
        """Run generation with Budget Forcing.

        Args:
            prompt: Input text (string). Should include the think_start
                marker at the end to begin the thinking phase.
            max_thinking_tokens: Thinking token budget. Default 2048.
            num_suppressions: How many times to suppress the end-of-thinking
                token, forcing the model to keep reasoning. Default 0.
            max_new_tokens: Maximum new tokens to generate. Default 1024.
            temperature: Sampling temperature. 0.0 = greedy. Default 0.0.

        Returns:
            dict with keys:
                full_output (str): Full decoded text (prompt + generation).
                thinking_tokens (int): Count of tokens in the thinking
                    portion (between think_start and think_end).
                answer (str): Extracted answer text.
                suppressions_used (int): How many times end-of-thinking was
                    suppressed, extending reasoning.
                forced_end (bool): Whether the end was forced by budget.
        """
        processor = BudgetForcingLogitsProcessor(
            tokenizer=self.tokenizer,
            think_end_str=self._think_end_str,
            max_thinking_tokens=max_thinking_tokens,
            num_suppressions=num_suppressions,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                logits_processor=[processor],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        full_output = self.tokenizer.decode(
            output_ids[0], skip_special_tokens=False
        )

        return {
            "full_output": full_output,
            "thinking_tokens": self._count_thinking_tokens(full_output),
            "answer": self._extract_answer(full_output),
            "suppressions_used": processor.suppressions_done,
            "forced_end": processor.force_end,
        }

    def _count_thinking_tokens(self, text):
        """Count thinking tokens between think_start and think_end markers.

        Searches for the markers in the decoded text and counts tokenizer
        tokens of the content between them. Returns 0 if either marker
        is missing.
        """
        start = text.find(self._think_start_str)
        end = text.find(self._think_end_str)
        if start == -1 or end == -1 or end <= start:
            return 0
        thinking_text = text[start + len(self._think_start_str) : end]
        return len(self.tokenizer.encode(thinking_text))

    def _extract_answer(self, text):
        """Extract answer content between think_end and answer_end markers.

        Returns everything after think_end_str up to (but not including)
        answer_end_str. Strips surrounding whitespace. Returns empty
        string if think_end marker is not found.
        """
        start = text.find(self._think_end_str)
        if start == -1:
            return ""
        end = text.find(self._answer_end_str, start)
        if end == -1:
            return text[start + len(self._think_end_str) :].strip()
        return text[start + len(self._think_end_str) : end].strip()
