from sklearn.model_selection import train_test_split

data_dir = "data/cs_data_test"
source_file = f"{data_dir}/full.source"
target_file = f"{data_dir}/full.target"

train_source_file = f"{data_dir}/train.source"
train_target_file = f"{data_dir}/train.target"
test_source_file = f"{data_dir}/test.source"
test_target_file = f"{data_dir}/test.target"
val_source_file = f"{data_dir}/val.source"
val_target_file = f"{data_dir}/val.target"


def write_lines(filename, lines):
    with open(filename, "w+") as f:
        for line in lines:
            f.write(line)

if __name__ == '__main__':

    # Read in all data
    with open(source_file) as f:
        full_source = f.readlines()

    with open(target_file) as f:
        full_target = f.readlines()

    # Create train-test-validation split
    train_source, test_source, train_target, test_target = train_test_split(
        full_source, full_target, test_size=0.2)

    train_source, val_source, train_target, val_target = train_test_split(
        train_source, train_target, test_size=0.5)

    # Write out splits to respective files
    write_lines(train_source_file, train_source)
    write_lines(train_target_file, train_target)

    write_lines(test_source_file, test_source)
    write_lines(test_target_file, test_target)

    write_lines(val_source_file, val_source)
    write_lines(val_target_file, val_target)

        