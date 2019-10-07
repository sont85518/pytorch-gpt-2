# pytorch-gpt-2



# tokenize data
python3 pregenerate_training_data_gpt2.py --model_name model/small --train_corpus aclImdb.txt --output_dir traindata  --num_file_lines 500000 --train_batch_size 32

# train gpt2
python3 run_gpt2.py --model_name model/small --do_train --do_eval --train_dataset traindata  --output_dir output_gpt --train_batch_size 32 --save_step 100 --gen_step 100


# sampling
python3 run_generation.py --model_type=gpt2 --length=200 --model_name_or_path=output_gpt/