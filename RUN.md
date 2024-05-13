export PYTHONPATH="/home/lucas/eccv/v-d4rl"

CUDA_VISIBLE_DEVICES=4

python scripts/real_pkl_to_hdf5.py --input_dir /localdata/lucas/py_projs/robo_dino/data/dataset/real_franka/eccvreb/button_press --output_dir /home/lucas/eccv/v-d4rl/vd4rl_data/main/button_press/expert

python conversion_scripts/npz_to_hdf5.py --input_dir  /home/lucas/eccv/v-d4rl/vd4rl_data/main/cheetah_run/expert/64px --output_dir /home/lucas/eccv/v-d4rl/vd4rl_data/main/cheetah_run/expert/raw_data

python drqbc/train.py task_name=offline_cheetah_run_expert offline_dir=vd4rl_data/main/cheetah_run/expert/viper_relabel algo=cql cql_importance_sample=false min_q_weight=10 seed=0


!pip install "opencv-python-headless<4.3"

CUDA_VISIBLE_DEVICES=1 python drqbc/train.py task_name=offline_cheetah_run_expert offline_dir=/home/lucas/eccv/v-d4rl/vd4rl_data/main/cheetah_run/expert/vqdiffusion_relabel algo=cql cql_importance_sample=false min_q_weight=10 seed=0