# ORDERS=('0,1,2,3' '0,1,3,2' '0,2,1,3' '0,2,3,1' '0,3,1,2' '0,3,2,1' '1,0,2,3' '1,0,3,2' '1,2,0,3' '1,2,3,0' '1,3,0,2' '1,3,2,0' '2,0,1,3' '2,0,3,1' '2,1,0,3' '2,1,3,0' '2,3,0,1' '2,3,1,0' '3,0,1,2' '3,0,2,1' '3,1,0,2' '3,1,2,0' '3,2,0,1' '3,2,1,0')
# ORDERS=('0,2,1,3')
ORDERS=('1,2,0,3')
 
for order in "${ORDERS[@]}"
do
  export ORDER=$order && \
  export RUN_NAME=d7e4b430_bs12_bf16_shard_over_batch_$ORDER && \
  echo $RUN_NAME && \
  python MaxText/inference_microbenchmark.py \
  MaxText/configs/base.yml \
  base_output_directory=gs://patemotter/maxtext-llama2-7b/microbenchmark \
  per_device_batch_size=12 \
  save_config_to_gcs=true \
  model_name=llama2-7b \
  tokenizer_path=assets/tokenizer.llama2 \
  max_prefill_predict_length=1024 \
  max_target_length=2048 \
  ici_fsdp_parallelism=1 \
  ici_tensor_parallelism=-1 \
  ici_autoregressive_parallelism=1 \
  weight_dtype=bfloat16 \
  enable_profiler=true \
  scan_layers=false \
  run_name=$RUN_NAME \
  ar_key_axis_order="$ORDER" \
  ar_value_axis_order="$ORDER" \
  prefill_key_axis_order="$ORDER" \
  prefill_value_axis_order="$ORDER" \
  > $RUN_NAME.out 2>&1
done
