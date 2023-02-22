python -m torch.distributed.launch --master_port=12347 --nproc_per_node=4 main.py something RGB \
     --arch tea50 --num_segments 8  \
     --gd 20 --lr 0.04 --lr_steps 30 40 45 --epochs 50 \
     --batch-size 32 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --experiment_name=TEA --shift \
     --shift_div=8 --shift_place=blockres --npb
