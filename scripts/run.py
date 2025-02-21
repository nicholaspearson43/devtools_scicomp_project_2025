from pyclassify.classifier import kNN
from pyclassify.utils import read_config, read_file
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Practical session 2")
parser.add_argument("--config", type=str, default=r"experiments/config")
args = parser.parse_args()

config_pth = args.config
#config_pth = fr"/home/nicholaspearson/DevTools/Lab01/devtools_scicomp_project_2025/experiments/config"
parameters = read_config(config_pth)

k = parameters["k"]
pth = parameters["dataset"]
try:
    backhand = parameters["backhand"]
except:
    backhand = "plain"

features, labels = read_file(pth)
n_obs = len(features)

indeces = np.arange(n_obs)
permuted_indeces = np.random.permutation(indeces)

split_idx = int(0.20*n_obs)
train_features = [features[i] for i in permuted_indeces[:split_idx]]
train_labels = [labels[i] for i in permuted_indeces[:split_idx]]
test_features = [features[i] for i in permuted_indeces[split_idx:]]
test_labels = [labels[i] for i in permuted_indeces[split_idx:]]

my_kNN = kNN(k, backhand)
y_pred = my_kNN.__call__((train_features, train_labels), test_features)
n_test = len(test_labels)

correct = 0
for obs in range(n_test):
    if y_pred[obs] == test_labels[obs]:
        correct += 1
accuracy = round(correct/n_test*100, 2)
print(f"Accuracy: {accuracy}%")