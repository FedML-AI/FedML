import asyncio
from typing import AsyncIterator, List, Sequence, TYPE_CHECKING, Union

import torch
from torch import Tensor
from transformers import TextStreamer

from .protocol import LLMOutput
from .typing import TokenizerType


# Adapted from `transformers.TextIteratorStreamer`
class AsyncTextIteratorStreamer(TextStreamer):
    def __init__(
            self,
            tokenizer: TokenizerType,
            skip_prompt: bool = False,
            stop: Union[str, Sequence[str]] = (),
            verbose: bool = False,
            **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)

        self.text_queue = asyncio.Queue()
        self.stop_signal = None
        self.stop_token_cache = []

        if not bool(stop):
            # if empty string or None
            stop = []
        elif isinstance(stop, str):
            stop = [stop]
        self.stop = [s for s in stop if len(s) > 0]

        # longest substring first
        self.stop_parts = sorted({s[:i + 1] for s in stop for i in range(len(s))}, key=len, reverse=True)

        # overwrite stop string check decode kwargs
        self.stop_check_decode_kwargs = self.decode_kwargs.copy()
        self.stop_check_decode_kwargs.update({"skip_special_tokens": False})

        # logging flag
        self.verbose = verbose

        if TYPE_CHECKING:
            self.tokenizer: TokenizerType = tokenizer

    def put(self, value: Tensor) -> None:
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if not self.next_tokens_are_prompt:
            new_token_ids = self.check_partial_stop(value.tolist())

            if len(new_token_ids) > 0:
                value = torch.tensor(new_token_ids, dtype=value.dtype, device=value.device)
            else:
                self.on_finalized_text("")
                return

        super().put(value)

        self.next_tokens_are_prompt = False

    def end(self) -> None:
        super().end()

        if self.verbose:
            print(
                f"[Stream End] ends with {self.tokenizer.decode(self.stop_token_cache, add_special_tokens=False)!r}",
                flush=True
            )
        self.stop_token_cache.clear()

    def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.text_queue.put_nowait(LLMOutput(
            text=text,
            # TODO: improve efficiency
            token_ids=self.tokenizer.encode(text, add_special_tokens=False)
        ))
        if stream_end:
            self.text_queue.put_nowait(self.stop_signal)

    def __aiter__(self) -> AsyncIterator[LLMOutput]:
        return self

    async def __anext__(self) -> LLMOutput:
        value = await self.text_queue.get()
        if value == self.stop_signal:
            raise StopAsyncIteration()
        else:
            return value

    def check_partial_stop(self, input_ids: List[int]) -> List[int]:
        token_ids = self.stop_token_cache + input_ids
        text = self.tokenizer.decode(token_ids, **self.stop_check_decode_kwargs)

        stop_str_start_idx = -1
        matched_stop_str = ""

        for s in self.stop:
            stop_str_start_idx = text.find(s)
            if stop_str_start_idx >= 0:
                matched_stop_str = text[stop_str_start_idx:]

                if self.verbose:
                    print(
                        f"[Stream] found full stop string {s!r} from {text[max(stop_str_start_idx - 10, 0):]!r}",
                        flush=True
                    )
                break

        else:
            for s in self.stop_parts:
                if text.endswith(s):
                    stop_str_start_idx = len(text) - len(s)
                    matched_stop_str = s

                    if self.verbose:
                        print(
                            f"[Stream] found partial stop string {s!r} from {text[max(stop_str_start_idx - 10, 0):]!r}",
                            flush=True
                        )
                    break

        if stop_str_start_idx >= 0 and bool(matched_stop_str):
            new_token_ids = matched_token_ids = None
            max_search_size = 0

            inferred_stop_token_ids = self.tokenizer.encode(matched_stop_str, add_special_tokens=False)
            matched_token_ids = token_ids[-len(inferred_stop_token_ids):]
            if len(matched_token_ids) > 0 and matched_token_ids == inferred_stop_token_ids:
                new_token_ids = token_ids[:-len(matched_token_ids)]
            else:
                max_search_size = max(max_search_size, len(matched_token_ids))

            if new_token_ids is None:
                matched_stop_ctx_str = f"\n{matched_stop_str}"
                inferred_stop_ctx_token_ids = self.tokenizer.encode(matched_stop_ctx_str, add_special_tokens=False)[2:]

                matched_token_ids = token_ids[-len(inferred_stop_ctx_token_ids):]
                if len(matched_token_ids) > 0 and matched_token_ids == inferred_stop_ctx_token_ids:
                    new_token_ids = token_ids[:-len(matched_token_ids)]
                else:
                    max_search_size = max(max_search_size, len(inferred_stop_ctx_token_ids))

            if new_token_ids is None:
                # find the longest suffix that matches the (partial) stop string
                cdd_strs = self.tokenizer.batch_decode(
                    [(i, token_ids[i:])
                     for i in range(max(len(token_ids) - max_search_size * 3, 0), len(token_ids))],
                    **self.decode_kwargs
                )

                for (start_idx, cdd_str) in cdd_strs:
                    if cdd_str == matched_stop_str:
                        new_token_ids = token_ids[:start_idx]
                        matched_token_ids = token_ids[start_idx:]
                        break

            assert new_token_ids is not None and matched_token_ids is not None
            self.stop_token_cache = matched_token_ids
            return new_token_ids

        else:
            self.stop_token_cache.clear()
            return token_ids
