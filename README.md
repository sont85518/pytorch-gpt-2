# pytorch-gpt-2

# sampling
1. Top-k sampling
2. Greedy
3. Beam search
4. Nucleus (The Curious Case of Neural Text Degeneration)

python3 pregenerate.py --model_name model/small --train_corpus data/debug_data.txt --output_dir data/train_data --num_file_lines 500000 --train_batch_size 32

python3 run.py --model_name model/small --do_train --do_eval --train_dataset data_movies_review/train_data  --output_dir output_gpt --train_batch_size 1 --save_step 100 --gen_step 100
