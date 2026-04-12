# !/bin/bash

# hyper-parameters sweep

# D-NGD
python main.py --dataset gsm8k --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --data_path /root/autodl-tmp --run_sweep --sweep_optimizer dngd --sweep_params '{"learning_rate":[3e-4,3e-5,3e-6,3e-7],"damping":[3e-1,3e-2,3e-3,3e-4,3e-5,6e-1,6e-2,6e-3,6e-4,6e-5],"momentum":[0.3,0.6,0.9],"weight_decay":[5e-4,5e-6,5e-2]}' --epoch 3 --batch_size 64 --eval_batch_size 64 --max_seq_length 512 --generation_max_new_tokens 256 --distributed_training True --num_workers 4 --gpu_numbers 4
# AdamW
python main.py --dataset gsm8k --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --data_path /root/autodl-tmp --run_sweep --sweep_optimizer adamw --sweep_params '{"learning_rate":[3e-4,3e-5,3e-6,3e-7],"betas":[(0.5, 0.55),(0.7,0.77),(0.9,0.99)],"eps":[1e-1,1e-2,1e-3,1e-4],"weight_decay":[5e-4,5e-6,5e-2]}' --epoch 3 --batch_size 64 --eval_batch_size 64 --max_seq_length 512 --generation_max_new_tokens 256 --distributed_training True --num_workers 4 --gpu_numbers 4
# Muon
python main.py --dataset gsm8k --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --data_path /root/autodl-tmp --run_sweep --sweep_optimizer muon --sweep_params '{"learning_rate":[3e-4,3e-5,3e-6,3e-7],"weight_decay":[5e-4,5e-6,5e-2],"eps":[1e-3,1e-5,1e-7,1e-9],"retraction_eps":[1e-3,1e-5,1e-7,1e-9]}' --epoch 3 --batch_size 64 --eval_batch_size 64 --max_seq_length 512 --generation_max_new_tokens 256 --distributed_training True --num_workers 4 --gpu_numbers 4
# SGD
python main.py --dataset gsm8k --network qwen3_0_6b --text_model_name Qwen/Qwen3-0.6B-Base --data_path /root/autodl-tmp --run_sweep --sweep_optimizer dngd --sweep_params '{"learning_rate":[3e-4,3e-5,3e-6,3e-7],"momentum":[0.3,0.6,0.9],"weight_decay":[5e-4,5e-6,5e-2]}' --epoch 3 --batch_size 64 --eval_batch_size 64 --max_seq_length 512 --generation_max_new_tokens 256 --distributed_training True --num_workers 4 --gpu_numbers 4

