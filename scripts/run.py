from pyclassify.classifier import kNN
from pyclassify.utils import read_config, read_file
import argparse

parser = argparse.ArgumentParser(description="Practical session 2")
parser.add_argument("--config", type=str, default=r"experiments/config")
args = parser.parse_args()

config_pth = args.config
#config_pth = fr"/home/nicholaspearson/DevTools/Lab01/devtools_scicomp_project_2025/experiments/config"
parameters = read_config(config_pth)

k = parameters["k"]
pth = parameters["dataset"]

features, labels = read_file(pth)
n_obs = len(features)

split_idx = int(0.20*n_obs)
train_features = features[:split_idx]
train_labels = labels[:split_idx]
test_features = features[split_idx:]
test_labels = labels[split_idx:]

my_kNN = kNN(k)
y_pred = my_kNN.__call__((train_features, train_labels), test_features)
n_test = len(test_labels)

correct = 0
for obs in range(n_test):
    if y_pred[obs] == test_labels[obs]:
        correct += 1
accuracy = round(correct/n_test*100, 2)
print(f"Accuracy: {accuracy}%")