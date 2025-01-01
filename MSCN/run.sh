# export CUDA_VISIBLE_DEVICES=1

/bin/python3 -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model MSCN --batch-size 2 --data-set UCM  --train_dir /home/amax/JW/dataset/UCM/train0.8 --test_dir /home/amax/JW/dataset/UCM/test0.2 --k1 2 --k2 6 --g 8