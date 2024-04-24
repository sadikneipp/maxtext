"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Provides op for tokenizing a dataset."""

from typing import Any, Dict, Iterable
import dataclasses
import tensorflow as tf
import tensorflow_text as tftxt
import max_logging
from jetstream.engine import tokenizer_api
from jetstream.third_party.llama3 import llama3_tokenizer


Features = Dict[str, tf.Tensor]


class TikToken(tokenizer_api.Tokenizer):
  """Tiktoken tokenizer"""
  def __init__(self, tokenizer_path: str, add_bos: bool, add_eos: bool):
    self.tokenizer = llama3_tokenizer.Tokenizer(tokenizer_path)
    self.add_bos = add_bos
    self.add_eos = add_eos

  def encode(self, s):
    t = []
    t.extend(self.tokenizer.model.encode(str(s)))
    if self.add_bos:
      t.insert(0, self.tokenizer.bos_id)
    if self.add_eos:
      t.append(self.tokenizer.eos_id)
    return t

  def decode(self, token_ids):
    return self.tokenizer.model.decode(token_ids)

  def decode_str(self, token_ids):
    # same as decode
    return self.decode(token_ids)

  @property
  def pad_id(self) -> int:
    """ID of the pad token."""
    return self.tokenizer.pad_id

  @property
  def eos_id(self) -> int:
    """ID of EOS token."""
    return self.tokenizer.eos_id

  @property
  def bos_id(self) -> int:
    """ID of the BOS token."""
    return self.tokenizer.bos_id

class SentencePieceTokenizer(tokenizer_api.Tokenizer):
  """SentencePiece tokenizer"""
  def __init__(self, tokenizer_path: str, add_bos: bool, add_eos: bool):
    with tf.io.gfile.GFile(tokenizer_path, "rb") as model_fp:
      sp_model = model_fp.read()
    self.sp_tokenizer = tftxt.SentencepieceTokenizer(model=sp_model, add_bos=add_bos, add_eos=add_eos)

  def encode(self, s):
    return self.sp_tokenizer.tokenize(s)

  def decode(self, token_ids):
    return self.sp_tokenizer.detokenize(token_ids)

  def decode_str(self, token_ids):
    # same as decode
    return self.decode(token_ids)

  @property
  def pad_id(self) -> int:
    """ID of the pad token."""
    return 0

  @property
  def eos_id(self) -> int:
    """ID of EOS token."""
    return self.sp_tokenizer.detokenize('</s>')

  @property
  def bos_id(self) -> int:
    """ID of the BOS token."""
    return self.sp_tokenizer.detokenize('<s>')


def build_tokenizer(tokenizer_path, add_bos, add_eos):
  """Loads the tokenizer at `tokenizer_path`"""
  max_logging.log(f"Tokenizer path: {tokenizer_path}")
  if "tiktoken" in tokenizer_path:
    return TikToken(tokenizer_path, add_bos, add_eos)
  return SentencePieceTokenizer(tokenizer_path, add_bos, add_eos)


@dataclasses.dataclass
class TokenizeOp:
  sp_tokenizer: Any
  data_keys: Iterable[str] = ("inputs", "targets")

  def __call__(self, features: Features) -> Features:
    for k in self.data_keys:
      features[k] = self.sp_tokenizer.encode(features[k])
    return features
