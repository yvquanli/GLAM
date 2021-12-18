import os
# os.chdir(os.path.dirname(__file__))

print('Test for dataset.py and run.py!')
os.system('python3 ./run.py --dataset demo --dataset_root ./demo --epoch 5 --loss bcel')
os.system('rm -rf ./log_demo')

print('Demo running...')
os.system('python3 ./glam.py --dataset demo --dataset_root ./demo --n_init_configs 5 --n_top_blend 2 --n_run_full_epoch 2')
