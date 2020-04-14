¡¾1. Environment Setup¡¿
Ubuntu 16.04, Python 3.6, Pytorch 0.4.1, CUDA 9.2, NVIDIA RTX 2080 GPUs

@ clone the repository
git clone https://github.com/foolwood/SiamMask.git && cd SiamMask
export SiamMask=$PWD

@ Setup python environment
conda create -n siammask python=3.6
source activate siammask
pip install -r requirements.txt
bash make.sh


@ Add the projcet to your PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

¡¾2. Deomo¡¿
@ Setup environment
@ Download SiamMask model
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth

@ Run demo.py
cd $SiamMask/experiments/siammask_sharp
export PYTHONPATH=$PWD:$PYTHONPATH
python ../../tools/demo.py --resume SiamMask_DAVIS.pth --config config_davis.json


¡¾3. Testing¡¿
@ Setup environment
cd $SiamMask/data

@ Download test data
sudo apt-get install jq
bash get_test_data.sh

@ Download pretrained models
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT_LD.pth
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth

@ Evaluate performance on VOT
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2016 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2018 0
bash test_mask_refine.sh config_vot.json SiamMask_VOT.pth VOT2019 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2016 0
bash test_mask_refine.sh config_vot18.json SiamMask_VOT_LD.pth VOT2018 0
python ../../tools/eval.py --dataset VOT2016 --tracker_prefix C --result_dir ./test/VOT2016
python ../../tools/eval.py --dataset VOT2018 --tracker_prefix C --result_dir ./test/VOT2018
python ../../tools/eval.py --dataset VOT2019 --tracker_prefix C --result_dir ./test/VOT2019


¡¾4. Training¡¿
@ Download data
Youtube-VOS, COCO, ImageNet-DET, ImageNet-VID

@ Preprocess each datasets according the 
https://github.com/foolwood/SiamMask/blob/master/data/coco/readme.md

@ Download the pre-trained model trained on ImageNet-1k Dataset
cd $SiamMask/experiments
wget http://www.robots.ox.ac.uk/~qwang/resnet.model
ls | grep siam | xargs -I {} cp resnet.model {}

£¡£¡4.1 £¡£¡ Training SiamMask base model
@ Setup Env
cd $SiamMask/experiments/siammask_base/
@ Run it
bash run.sh

# 10hours more than
# reduce batch size in run.sh, to get out of out-of-memory errors
# View progress in Tensorboard (logs are at <experiment_dir>/logs/)

@ After training, test
bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4  # test all snapshots with 4 GPUs

£¡£¡4.2 £¡£¡Training SiamMask Model with Refine Module
@ Setup environment
cd $SiamMask/experiments/siammask_sharp

@ train with the best SiamMask base Model
bash run.sh <best_base_model>
bash run.sh checkpoint_e12.pth

# You can view progress on Tensorboard (logs are at <experiment_dir>/logs/)

@ After training, you can test checkpoints on VOT dataset
bash test_all.sh -s 1 -e 20 -d VOT2018 -g 4



