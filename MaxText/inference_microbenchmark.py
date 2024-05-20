"""
Copyright 2024 Google LLC

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

"""Inference microbenchmark for prefill and autoregressive steps."""
import datetime
import jax
import json
import sys

from jetstream.engine import token_utils

import max_utils
import maxengine
import maxtext_utils
import pyconfig


_WARMUP_ITERS = 2


def prefill_benchmark_loop(engine, params, tokens, true_length, iters):
  """Inner loop for benchmarking prefill step."""
  start = datetime.datetime.now()
  for _ in range(iters):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
  jax.block_until_ready(prefill_result)
  end = datetime.datetime.now()
  max_utils.delete_pytree(prefill_result)
  return (end - start).total_seconds()


def prefill_benchmark(
    config, engine, params, tokens, true_length, num_model_params, iters
):
  """Handles warmup, running prefill benchmark, and printing results."""
  for _ in range(_WARMUP_ITERS):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
  jax.block_until_ready(prefill_result)
  max_utils.delete_pytree(prefill_result)

  print(f"Prefill benchmark results for length {tokens.size}:\n")
  time_in_s = prefill_benchmark_loop(engine, params, tokens, true_length, iters)
  prefill_average_ms = 1000 * time_in_s / iters
  prefill_tflops_per_device, _, _ = maxtext_utils.calculate_prefill_tflops_per_device(num_model_params, tokens.size, config)
  tflops_per_sec_per_device = prefill_tflops_per_device /  prefill_average_ms * 1000.0
  print(
      f"\tPrefill step average time: {prefill_average_ms:.3f} ms\n"
      f"\tPrefill total TFLOPs/device: {prefill_tflops_per_device:.3f}\n"
      f"\tPrefill TFLOPs/sec/device: {tflops_per_sec_per_device:.3f}\n\n\n\n"
  )
  result_dict = {
      "prefill_time_in_ms": prefill_average_ms,
      "prefill_total_tflops_per_device": prefill_tflops_per_device,
      "prefill_tflops_per_sec_per_device": tflops_per_sec_per_device,
  }
  return result_dict


def prefill_insert_benchmark_loop(
    config, engine, decode_state, params, total_slots, tokens, true_length, iters, profile_name
  ):
  """Inner loop for benchmarking prefill and insert step."""
  max_utils.activate_profiler(config, profile_name)
  start = datetime.datetime.now()
  for i in range(iters):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = engine.insert(prefill_result, decode_state, int(i % total_slots))
    max_utils.delete_pytree(prefill_result)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  max_utils.deactivate_profiler(config)
  return (end - start).total_seconds(), decode_state


def prefill_insert_benchmark(
    config, engine, decode_state, params, total_slots, tokens, true_length, iters
  ):
  """Handles warmup, running insert benchmark, and printing results."""

  for i in range(_WARMUP_ITERS):
    prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
    decode_state = engine.insert(prefill_result, decode_state, int(i % total_slots))
    max_utils.delete_pytree(prefill_result)
  jax.block_until_ready(decode_state)

  print(f"Prefill and insert benchmark results for length {tokens.size}:\n")
  time_in_s, decode_state = prefill_insert_benchmark_loop(
    config, engine, decode_state, params, total_slots, tokens, true_length, iters, f"prefill_insert_{tokens.size}")
  prefill_insert_average_ms = time_in_s / iters * 1000.0
  print(
      f"\tPrefill + Insert step average time: {prefill_insert_average_ms:.3f} ms\n\n\n\n"
  )
  result_dict = {
      "prefill_insert_time_in_ms": prefill_insert_average_ms
  }
  return result_dict, decode_state


def ar_benchmark_loop(config, engine, params, decode_state, iters, profile_name):
  """Inner loop for benchmarking ar step."""
  max_utils.activate_profiler(config, profile_name)
  start = datetime.datetime.now()
  for _ in range(iters):
    decode_state, _ = engine.generate(params, decode_state)
  jax.block_until_ready(decode_state)
  end = datetime.datetime.now()
  max_utils.deactivate_profiler(config)
  return (end - start).total_seconds(), decode_state


def ar_benchmark(config, engine, params, decode_state, global_batch_size, cache_size, model_size, iters):
  """Handles warmup, running ar benchmark, and printing results."""
  for _ in range(_WARMUP_ITERS):
    decode_state, _ = engine.generate(params, decode_state)
  jax.block_until_ready(decode_state)

  time_in_s, decode_state = ar_benchmark_loop(config, engine, params, decode_state, iters, profile_name="autoregress")
  seconds_per_step = time_in_s / iters
  ar_average_ms = seconds_per_step * 1000
  total_throughput = global_batch_size / seconds_per_step

  GB_per_step_per_device = (model_size + cache_size) / 1e9 / jax.device_count()
  bw_per_device = GB_per_step_per_device / seconds_per_step
  print(
      f"AutoRegressive results:\n"
      f"\tAR step average time: {ar_average_ms:.3f} ms\n"
      f"\tAR step average time per seq: {ar_average_ms/global_batch_size:.3f} ms\n"
      f"\tAR global batch size: {global_batch_size}\n"
      f"\tAR throughput: {total_throughput:.3f} tokens/second\n"
      f"\tAR memory bandwidth per device: {bw_per_device:.3f} GB/s\n\n\n"
  )

  result_dict = {
      "ar_step_in_ms": ar_average_ms,
      "ar_step_in_ms_per_seq": ar_average_ms / global_batch_size,
      "ar_global_batch_size": global_batch_size,
      "ar_total_throughput_tokens_per_second": total_throughput,
      "ar_device_bandwidth_GB_per_second": bw_per_device,
  }
  return result_dict, decode_state


def collate_results(config, results, model_size, cache_size, num_model_params, incl_config=False):
  """Adds model/cache size info and optionally config info to results."""
  results["sizes"] = {
      "Model_size_in_GB": model_size / 1e9,
      "cache_size_in_GB": cache_size / 1e9,
      "model_params_in_billions": num_model_params / 1e9,
  }
  if incl_config:
    results["config"] = {}
    for k, v in dict(config.get_keys()).items():
      results["config"][k] = str(v) if k == "dtype" else v  # json fails with original dtype
  return results


def write_results(results, filename):
  if filename != "":
    with open(filename, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2)


def print_results_for_analyze(results):
  """Print results."""
  print("\nFor usage in analyze_sharegpt.py :")

  if "Prefill" in results:
    prefill_bucket_size_to_ms = {}
    for k, v in results["Prefill"].items():
      prefill_bucket_size_to_ms[int(k)] = round(v["prefill_time_in_ms"], 3)
    print(f"PREFILL_BUCKET_SIZE_TO_MS = {prefill_bucket_size_to_ms}")

  if "Prefill_Insert" in results:
    insert_bucket_size_to_ms = {}
    for k, v in results["Prefill_Insert"].items():
      insert_bucket_size_to_ms[int(k)] = round(v["prefill_insert_time_in_ms"], 3)
    print(f"PREFILL_INSERT_BUCKET_SIZE_TO_MS = {insert_bucket_size_to_ms}")

  if "AutoRegressive" in results:
    print(f"SYSTEM_TIME_PER_DECODE_TOKEN_MS = {results['AutoRegressive']['ar_step_in_ms_per_seq']}")


def summarize_prefill_result(engine, params, tokens, true_length):
  """Summarize Prefill result."""
  print(f"Prefill result of length {tokens.size}:\n")
  prefill_result = engine.prefill(params=params, padded_tokens=tokens, true_length=true_length)
  jax.block_until_ready(prefill_result)
  num_prefill_logits_params, total_prefill_logits_size, avg_prefill_logits_param_size = (
    max_utils.summarize_pytree_data(prefill_result["logits"], name="Prefill Logits", raw=True)
  )
  num_prefill_cache_params, total_prefill_cache_size, avg_prefill_cache_param_size = (
    max_utils.summarize_pytree_data(prefill_result["cache"], name="Prefill Cache")
  )
  max_utils.delete_pytree(prefill_result)
  return {
    "num_prefill_logits_params": num_prefill_logits_params,
    "total_prefill_logits_size": total_prefill_logits_size,
    "avg_prefill_logits_param_size": avg_prefill_logits_param_size,
    "num_prefill_cache_params": num_prefill_cache_params,
    "total_prefill_cache_size": total_prefill_cache_size,
    "avg_prefill_cache_param_size": avg_prefill_cache_param_size,
  }


def main(config):
  engine = maxengine.MaxEngine(config)
  params = engine.load_params()
  prefill_lengths = [int(l) for l in config.inference_microbenchmark_prefill_lengths.split(",")]
  stages_to_benchmark = config.inference_microbenchmark_stages.split(",")
  benchmark_loop_iters = config.inference_microbenchmark_loop_iters

  text = config.prompt
  metadata = engine.get_tokenizer()
  vocab = token_utils.load_vocab(metadata.path, metadata.extra_ids)

  decode_state = engine.init_decode_state()
  _, cache_size, _ = max_utils.summarize_pytree_data(decode_state["cache"], name="Cache")
  num_model_params, model_size, _ = max_utils.summarize_pytree_data(params, name="Model")

  benchmark_results = {}
  if "prefill" in stages_to_benchmark:

    benchmark_results["Prefill_Result"] = {}
    benchmark_results["Prefill"] = {}
    benchmark_results["Prefill_Insert"] = {}
    prefill_tokens = {}
    prefill_true_lengths = {}

    for prefill_length in prefill_lengths:
      prefill_tokens[prefill_length], prefill_true_lengths[prefill_length] = token_utils.tokenize_and_pad(
        text, vocab, is_bos=True, prefill_lengths=[prefill_length]
      )
      benchmark_results["Prefill_Result"]["prefill_length"] = summarize_prefill_result(
        engine, params, prefill_tokens[prefill_length], prefill_true_lengths[prefill_length]
      )

    for prefill_length in prefill_lengths:
      benchmark_results["Prefill"][prefill_length] = prefill_benchmark(
        config,
        engine,
        params,
        prefill_tokens[prefill_length],
        prefill_true_lengths[prefill_length],
        num_model_params,
        benchmark_loop_iters
      )

      benchmark_results["Prefill_Insert"][prefill_length], decode_state = prefill_insert_benchmark(
        config,
        engine,
        decode_state,
        params,
        engine.max_concurrent_decodes,
        prefill_tokens[prefill_length],
        prefill_true_lengths[prefill_length],
        benchmark_loop_iters
      )

  if "generate" in stages_to_benchmark:

    import inference_utils
    import common_types
    from jax import numpy as jnp
    from jax._src.layout import Layout, DeviceLocalLayout as DLL
    from flax.linen import partitioning as nn_partitioning
    from jetstream.engine import engine_api
    import re

    pattern = re.compile(r"\{(.*?):")

    # Extract minor_to_major from str(layout) because layout doesn't have a
    # minor_to_major property yet.
    def extract_minor_to_major(l):
      match = re.search(pattern, str(l))
      if match:
        return tuple(int(i) for i in match.groups()[0].split(','))
      return l

    def modify_pytree_layout(p):
      def modify_leaf_layout(leaf):
        if isinstance(leaf, Layout):
          leaf.device_local_layout = DLL.AUTO
      jax.tree_util.tree_map(modify_leaf_layout, p)

    def modify_pytree_sharding(p):
      def modify_leaf_layout(leaf):
        if isinstance(leaf, Layout):
          leaf.sharding = None
      jax.tree_util.tree_map(modify_leaf_layout, p)

    def generate(params, decode_state):
      """Run one generate step"""
      previous_logits = decode_state["logits"]

      new_token = inference_utils.sampling(
          previous_logits,
          engine.rng,
          engine.config.decode_sampling_strategy,
          topk=engine.config.decode_sampling_top_k,
          nucleus_topp=engine.config.decode_sampling_nucleus_p,
          temperature=engine.config.decode_sampling_temperature,
      )

      with engine.mesh, nn_partitioning.axis_rules(engine.config.logical_axis_rules):
        out_logits, new_vars = engine.model.apply(
            params | {"cache": decode_state["cache"]},
            new_token,
            decode_state["next_pos"],
            enable_dropout=False,
            model_mode=common_types.MODEL_MODE_AUTOREGRESSIVE,
            rngs={"params": engine.rng},
            mutable=["cache"],
        )

      all_valid = jnp.ones(new_token.shape, dtype=jnp.int8)

      result = engine_api.ResultTokens(
          data=jnp.concatenate((new_token, all_valid, decode_state["generated_tokens"]), axis=1),
          # Tokens are shape [batch, speculations], so when we concatenate
          # tokens, validity and length along their index 1 dimension then they
          # occupy 0:speculations.
          tokens_idx=(0, 1),
          # Validity occupies the same amount of space, but next in line.
          valid_idx=(1, 2),
          # And lengths is rank 1.
          length_idx=(2, 3),
          samples_per_slot=1,
      )

      out_logits = jax.lax.with_sharding_constraint(out_logits, engine.replicated_sharding)
      new_cache = jax.lax.with_sharding_constraint(new_vars["cache"], engine.kv_cache_shardings)

      return {
          "logits": out_logits,
          "cache": new_cache,
          "next_pos": decode_state["next_pos"] + 1,
          "generated_tokens": decode_state["generated_tokens"] + 1,
      }, result
    
    def get_generate_layouts(compiled_generate):
      (params_layout, decode_state_layout), _ = compiled_generate.input_layouts()
      (generated_decode_state_layout, result_layout) = compiled_generate.output_layouts()
      return params_layout, decode_state_layout, generated_decode_state_layout, result_layout

    def inspect_decode_state_layouts(decode_state_layout, prefix):
      attention_op_0 = decode_state_layout["cache"]["decoder"]['layers_0']['self_attention']['AttentionOp_0']
      cached_prefill_key_layout = attention_op_0['cached_prefill_key']
      cached_prefill_value_layout = attention_op_0['cached_prefill_value']
      cached_ar_key_layout = attention_op_0['cached_ar_key']
      cached_ar_value_layout = attention_op_0['cached_ar_value']
      print(f"{prefix} decode_state_layouts: ")
      print(f"    cached_prefill_key_layout: {extract_minor_to_major(cached_prefill_key_layout)}")
      print(f"    cached_prefill_value_layout: {extract_minor_to_major(cached_prefill_value_layout)}")
      print(f"    cached_ar_key_layout: {extract_minor_to_major(cached_ar_key_layout)}")
      print(f"    cached_ar_value_layout: {extract_minor_to_major(cached_ar_value_layout)}")

    abstract_params = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding), params)
    abstract_decode_state = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding), decode_state)

    # Concrete
    concrete_lowered_generate = jax.jit(generate).lower(params, decode_state)
    concrete_compiled_generate = concrete_lowered_generate.compile()
    (
      concrete_params_layout,
      concrete_decode_state_layout,
      concrete_generated_decode_state_layout,
      concrete_result_layout
    ) = get_generate_layouts(concrete_compiled_generate)
    inspect_decode_state_layouts(concrete_decode_state_layout, "concrete")

    # Optimized with Sharding
    modify_pytree_layout(concrete_params_layout)
    modify_pytree_layout(concrete_decode_state_layout)
    modify_pytree_layout(concrete_generated_decode_state_layout)
    modify_pytree_layout(concrete_result_layout)
    inspect_decode_state_layouts(concrete_decode_state_layout, "modified to default")
    optimized_w_sharding_lowered_generate = jax.jit(
      generate,
      in_shardings=(concrete_params_layout, concrete_decode_state_layout),
      out_shardings=(concrete_generated_decode_state_layout, concrete_result_layout)
    ).lower(abstract_params, abstract_decode_state)
    optimized_w_sharding_compiled_generate = optimized_w_sharding_lowered_generate.compile()
    (
      optimized_w_sharding_params_layout,
      optimized_w_sharding_decode_state_layout,
      optimized_w_sharding_generated_decode_state_layout,
      optimized_w_sharding_result_layout
    ) = get_generate_layouts(optimized_w_sharding_compiled_generate)
    inspect_decode_state_layouts(optimized_w_sharding_decode_state_layout, "optimized with sharding")

    # Optimized without Sharding
    modify_pytree_sharding(concrete_params_layout)
    modify_pytree_sharding(concrete_decode_state_layout)
    modify_pytree_sharding(concrete_generated_decode_state_layout)
    modify_pytree_sharding(concrete_result_layout)
    optimized_wo_sharding_lowered_generate = jax.jit(
      generate,
      in_shardings=(concrete_params_layout, concrete_decode_state_layout),
      out_shardings=(concrete_generated_decode_state_layout, concrete_result_layout)
    ).lower(abstract_params, abstract_decode_state)
    optimized_wo_sharding_compiled_generate = optimized_wo_sharding_lowered_generate.compile()
    (
      optimized_wo_sharding_params_layout,
      optimized_wo_sharding_decode_state_layout,
      optimized_wo_sharding_generated_decode_state_layout,
      optimized_wo_sharding_result_layout
    ) = get_generate_layouts(optimized_wo_sharding_compiled_generate)
    inspect_decode_state_layouts(optimized_wo_sharding_decode_state_layout, "optimized without sharding")

    benchmark_results["AutoRegressive"], decode_state = ar_benchmark(
      config, engine, params, decode_state, engine.max_concurrent_decodes, cache_size, model_size, benchmark_loop_iters)

  results = collate_results(config, benchmark_results, model_size, cache_size, num_model_params)
  write_results(results, filename=config.inference_microbenchmark_log_file_path)
  print_results_for_analyze(results)


if __name__ == "__main__":
  pyconfig.initialize(sys.argv)
  main(pyconfig.config)
