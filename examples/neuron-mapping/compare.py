import torch
import matplotlib.pyplot as plt
import numpy as np

from taker.activations import get_top_frac


def compare_pruned_ff_criteria(
    cripple_repos: list[str],
    model_size: str,
    path: str = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/",
    focus_repo: str = "pile",
    prune_ratio: float = 0.01,
):
    # cripple_repos = ["physics", "bio", "code"]
    directory = f"{path}{model_size}/"
    suffix = f"-{model_size}-0.01-recent.pt"
    ratios = {}
    ratios["model_size"] = model_size

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for repo1 in cripple_repos:
        # load ff_criteria from repo1
        repo1_tensors = torch.load(
            directory + repo1 + "-" + focus_repo + suffix,
            map_location=torch.device(device),
        )
        # calculate ff_criteria for repo1 from the neuron scores
        repo1_ff_scores = repo1_tensors["ff_scores"].float()
        repo1_ff_criteria, _ = get_top_frac(repo1_ff_scores, prune_ratio)

        ratios[repo1] = {}
        for repo2 in cripple_repos:
            if repo1 == repo2:
                ratios[repo1][repo2] = 0
            # load ff_criteria from repo2
            repo2_tensors = torch.load(
                directory + repo2 + "-" + focus_repo + suffix,
                map_location=torch.device(device),
            )
            # calculate ff_criteria for repo2 from the neuron scores
            repo2_ff_scores = repo2_tensors["ff_scores"].float()
            repo2_ff_criteria, _ = get_top_frac(repo2_ff_scores, prune_ratio)

            matches = torch.logical_and(repo1_ff_criteria, repo2_ff_criteria)
            ratio = torch.sum(matches) / torch.sum(repo1_ff_criteria)
            ratios[repo1][repo2] = ratio

    return ratios


datasets = [
    f"cifar20-{dataset}"
    for dataset in [
        "aquatic_mammals",
        "fish",
        "flowers",
        "food_containers",
        "fruit_and_vegetables",
        "household_electrical_devices",
        "household_furniture",
        "insects",
        "large_carnivores",
        "large_natural_outdoor_scenes",
        "large_omnivores_and_herbivores",
        "medium_sized_mammals",
        "non_insect_invertebrates",
        "people",
        "reptiles",
        "small_mammals",
        "trees",
        "veh1",
        "veh2",
    ]
]

prune_ratio = 0.01

comparison = compare_pruned_ff_criteria(
    datasets,
    "init",
    path="examples/neuron-mapping/saved_tensors/",
    focus_repo="cifar20-split",
    prune_ratio=prune_ratio,
)

grid = [
    [
        comparison[dataset_a][dataset_b].item() if dataset_a != dataset_b else np.nan
        for dataset_b in datasets
    ]
    for dataset_a in datasets
]
grid = np.ma.masked_where(np.isnan(grid), grid)

average = np.mean(grid)

# reset all plot settings from configuration in taker proper
plt.rcdefaults()

plt.imshow(grid)
for i in range(len(datasets)):
    for j in range(len(datasets)):
        plt.text(
            j,
            i,
            f"{grid[i, j]:.2f}",
            ha="center",
            va="center",
            color="black" if grid[i, j] > average else "white",
        )
plt.xticks(
    range(len(datasets)), [dataset.split("-")[1] for dataset in datasets], rotation=90
)
plt.yticks(range(len(datasets)), [dataset.split("-")[1] for dataset in datasets])

plt.subplots_adjust(bottom=0.3)

plt.title(f"FF Criteria Overlap for Prune Ratio {prune_ratio}")

plt.show()
