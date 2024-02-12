import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def compare_pruned_ff_criteria(cripple_repos: list[str], model_size: str, path: str="/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/", focus_repo: str = "pile",):
    # cripple_repos = ["physics", "bio", "code"]
    directory = f"{path}{model_size}/"
    suffix = "-"+model_size+"-0.01-recent.pt"
    ratios = {}
    ratios["model_size"] = model_size
    
    for repo1 in cripple_repos:
        #load ff_criteria from repo1
        repo1_tensors = torch.load(directory+repo1+"-"+focus_repo+suffix)
        repo1_ff_criteria = repo1_tensors["ff_criteria"]
        ratios[repo1] = {}
        for repo2 in cripple_repos:
            if repo1 == repo2:
                ratios[repo1][repo2] = 0
            #load ff_criteria from repo2
            repo2_tensors = torch.load(directory+repo2+"-"+focus_repo+suffix)
            repo2_ff_criteria = repo2_tensors["ff_criteria"]


            matches = torch.logical_and(repo1_ff_criteria, repo2_ff_criteria)
            ratio = torch.sum(matches)/torch.sum(repo1_ff_criteria)
            ratios[repo1][repo2] = ratio
            
    return ratios

cripple_datasets = [f"cifar20-{dataset}" for dataset in ["aquatic_mammals", "fish", "flowers", "food_containers",
                    "fruit_and_vegetables", "household_electrical_devices", "household_furniture", "insects",
                    "large_carnivores", "large_natural_outdoor_scenes", "large_omnivores_and_herbivores",
                    "medium_sized_mammals", "non_insect_invertebrates", "people", "reptiles", "small_mammals",
                    "trees", "veh1", "veh2"]]
    
pruned_comparison = compare_pruned_ff_criteria(cripple_datasets, "Cifar100", path="examples/neuron-mapping/saved_tensors/",
                                               focus_repo="cifar20-split")

# plot with imshow
formatted_pruned_comparison = np.array([[pruned_comparison[repo1][repo2] for repo2 in cripple_datasets] for repo1 in cripple_datasets])

# masked_array = np.ma.array(formatted_pruned_comparison, mask=np.isnan(formatted_pruned_comparison))
# cmap = matplotlib.colormaps.get_cmap('viridis')
# cmap.set_bad('white',1.)
plt.imshow(formatted_pruned_comparison, interpolation='nearest')

# add labels
plt.xticks(range(len(cripple_datasets)), cripple_datasets, rotation=90)
plt.yticks(range(len(cripple_datasets)), cripple_datasets)

# add numbers to each cell
for i in range(len(cripple_datasets)):
    for j in range(len(cripple_datasets)):
        plt.text(j, i, f"{formatted_pruned_comparison[i, j]:.2f}", ha='center', va='center', color='white')

plt.show()
