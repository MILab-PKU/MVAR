CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 12347 \
autoregressive/sample/sample_icam2i_nv_ray_ddp_gso.py \
--t5-path pretrained_models/ \
--t5-model-type flan-t5-xl \
--gpt-ckpt pretrained_models/image_condition.pt \
--cfg 1.5 --kvcache --num-views 4
