import pickle

from train_bert import preprocessing_for_bert, bert_predict
from split_data import load_dfs

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

models_dir = "models/tutorial"
trained_classifier_path = f"{models_dir}/ft_bert.pickle"

if __name__ == '__main__':

    # Load fine-tuned model from train_bert.py
    with open(trained_classifier_path, 'rb') as f:
        bert_classifier = pickle.load(f)

    X_train, X_val, y_train, y_val, X, y, data, test_data = load_dfs()

    print(test_data.sample(5))

    # Create the DataLoader for our test set
    test_dataset = TensorDataset(test_inputs, test_masks)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

    # Compute predicted probabilities on the test set
    probs = bert_predict(bert_classifier, test_dataloader)

    # Get predictions from the probabilities
    threshold = 0.9
    preds = np.where(probs[:, 1] > threshold, 1, 0)

    # Number of tweets predicted non-negative
    print("Number of tweets predicted non-negative: ", preds.sum())

    output = test_data[preds==1]
    print(list(output.sample(20).tweet))
