"""
    Create a small subset of the training data for testing / debugging purposes. More convenient because run time is significantly cut down
"""

from sklearn.model_selection import train_test_split


train_data_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/training_data/cs_train_data_w2v"
out_dir = "/home/aukey2/gen-cs-wiki-articles/rag_branch/data/training_data/cs_train_data_w2v_subset"



def save_subset(task, perc=.02):
    """
    Arguments:
    - task: one of {'train', 'test', 'val'}
    - perc: percentage of data to include in subset
    """
    task_srcs = []

    src_file = task + '.source'
    trg_file = task + '.target'

    with open(train_data_dir + '/' + src_file) as f:
        src_lines = f.readlines()
    
    with open(train_data_dir + '/' + trg_file) as f:
        trg_lines = f.readlines()

    _, src_lines, _, trg_lines = train_test_split(src_lines, trg_lines, test_size=perc)


    with open(out_dir + '/' + src_file, "w+") as f:
        for l in src_lines:
            f.write(l)

    with open(out_dir + '/' + trg_file, "w+") as f:
        for l in trg_lines:
            f.write(l)


save_subset("train", perc=0.001)
save_subset("test")
save_subset("val")
