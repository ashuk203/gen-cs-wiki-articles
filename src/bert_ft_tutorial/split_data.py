import pandas as pd
from sklearn.model_selection import train_test_split



def load_dfs():
    # Load data and set labels
    data_complaint = pd.read_csv('data/complaint1700.csv')
    data_complaint['label'] = 0
    data_non_complaint = pd.read_csv('data/noncomplaint1700.csv')
    data_non_complaint['label'] = 1

    # Concatenate complaining and non-complaining data
    data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)

    # Drop 'airline' column
    data.drop(['airline'], inplace=True, axis=1)

    # Display 5 random samples
    print(data.sample(5))


    X = data.tweet.values
    y = data.label.values

    X_train, X_val, y_train, y_val =\
        train_test_split(X, y, test_size=0.1, random_state=2020)


    # Load test data
    test_data = pd.read_csv('data/test_data.csv')

    # Keep important columns
    test_data = test_data[['id', 'tweet']]

    # Display 5 samples from the test data
    print(test_data.sample(5))

    return X_train, X_val, y_train, y_val, X, y, data, test_data