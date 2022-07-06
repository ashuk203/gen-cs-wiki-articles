import re

sent_end_pattern = re.compile("\.")

def get_first_sent(par):
    match = re.search(sent_end_pattern, par)

    if not match:
        return ""

    first_sent = par[:match.start() + 1]
    return first_sent

src_train_data_dir = "data/cs_train_data"
dst_train_data_dir = "data/cs_train_first_sent"


def strip_def(file_name):
    src_file = f"{src_train_data_dir}/{file_name}"
    dst_file = f"{dst_train_data_dir}/{file_name}"

    with open(src_file) as f_in, open(dst_file, "w+") as f_out:
        for line in f_in:
            def_delim_idx = line.index(':')

            keyword = line[:def_delim_idx]
            summary = line[def_delim_idx + 1:]
            first_sent = get_first_sent(summary)

            if len(first_sent) == 0:
                continue

            new_train_ln = keyword + ': ' + first_sent 
            f_out.write(new_train_ln + '\n')

if __name__ == '__main__':
    strip_def("test.source")
    strip_def("train.source")
    strip_def("val.source")

    # cp data/cs_train_data/test.target data/cs_train_data_no_inp_def/test.target
    # cp data/cs_train_data/train.target data/cs_train_data_no_inp_def/train.target
    # cp data/cs_train_data/val.target data/cs_train_data_no_inp_def/val.target
