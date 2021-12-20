mkdir logs

conda create -n simpletransformers python pandas tqdm
conda activate simpletransformers
# conda install pytorch cudatoolkit=10.2 -c pytorch

pip install -r requirements.txt