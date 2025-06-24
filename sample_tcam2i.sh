CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node 1 --master-port 12345 \
autoregressive/sample/sample_tcam2i_nv_ray_ddp.py \
--t5-path pretrained_models/ \
--t5-model-type flan-t5-xl \
--gpt-ckpt pretrained_models/text_condition.pt \
--prompt-pkl dataset/pkls/objaverse_test_ids.pkl \
--cfg 1.5 --kvcache --num-views 4 \
--image-size 256
