import torch
import matplotlib.pyplot as plt
import numpy as np 

def compare_pruned_ff_criteria(cripple_repos: list[str], model_size: str, path: str="/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/", focus_repo: str = "pile", prune_ratio: float = 0.01):
    # cripple_repos = ["physics", "bio", "code"]
    directory = f"{path}{model_size}/"
    suffix = f"-{model_size}-{prune_ratio}-recent.pt"
    ratios = {}
    ratios["model_size"] = model_size

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for repo1 in cripple_repos:
        #load ff_criteria from repo1
        repo1_tensors = torch.load(directory+repo1+"-"+focus_repo+suffix, map_location=torch.device(device))
        repo1_ff_criteria = repo1_tensors["ff_criteria"]
        ratios[repo1] = {}
        for repo2 in cripple_repos:
            if repo1 == repo2:
                continue
            #load ff_criteria from repo2
            repo2_tensors = torch.load(directory+repo2+"-"+focus_repo+suffix, map_location=torch.device(device))
            repo2_ff_criteria = repo2_tensors["ff_criteria"]


            matches = torch.logical_and(repo1_ff_criteria, repo2_ff_criteria)
            ratio = torch.sum(matches)/torch.sum(repo1_ff_criteria)
            ratios[repo1][repo2] = ratio
            
    return ratios

datasets = ["cifar20-trees", "cifar20-veh1", "cifar20-veh2"]

comparison = compare_pruned_ff_criteria(datasets, "Cifar100", path="examples/neuron-mapping/saved_tensors/", focus_repo="cifar20-split")
print(comparison)

grid = [[comparison[dataset_a][dataset_b].item() if dataset_a != dataset_b else np.nan for dataset_b in datasets] for dataset_a in datasets]
grid = np.ma.masked_where(np.isnan(grid), grid)

average = np.mean(grid)

plt.imshow(grid)
for i in range(len(datasets)):
    for j in range(len(datasets)):
        plt.text(j, i, f"{grid[i, j]:.2f}", ha='center', va='center', color='black' if grid[i,j] > average else 'white')
plt.xticks(range(len(datasets)), datasets)
plt.yticks(range(len(datasets)), datasets)
plt.show()