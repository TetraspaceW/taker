
from taker.data_classes import PruningConfig
from taker.prune import run_pruning
import torch

def compare_pruned_ff_criteria(cripple_repos: list[str], model_size: str):
    directory = "/home/ubuntu/taker-rashid/examples/neuron-mapping/saved_tensors/"+model_size+"/"
    focus_repo = "pile"
    suffix = "-"+model_size+"-recent.pt"
    ratios = {}
    ratios["model_size"] = model_size
    
    for repo1 in cripple_repos:
        #load ff_criteria from repo1
        repo1_tensors = torch.load(directory+repo1+"-"+focus_repo+suffix)
        repo1_ff_criteria = repo1_tensors["ff_criteria"]
        ratios[repo1] = {}
        for repo2 in cripple_repos:
            if repo1 == repo2:
                continue
            #load ff_criteria from repo2
            repo2_tensors = torch.load(directory+repo2+"-"+focus_repo+suffix)
            repo2_ff_criteria = repo2_tensors["ff_criteria"]

            matches = torch.logical_and(repo1_ff_criteria, repo2_ff_criteria)
            ratio = torch.sum(matches)/torch.sum(repo1_ff_criteria)
            ratios[repo1][repo2] = ratio
            
    return ratios 
    
def get_shared_pruning_data(
        model_repo: str = "nickypro/tinyllama-15M",
        model_size: str = None,
        cripple_repos: list[str] = ["physics", "biology","chemistry", "math", "code"],
        focus_repo: str = "pile"
    ):
    
    # Configure initial model and tests
    c = PruningConfig(
        wandb_project   = "testing", # repo to push results to
        model_repo  = model_repo, # the model to prune
        token_limit = 1000,  # trim the input to this max length
        run_pre_test    = False, # evaluate the unpruned model
        eval_sample_size    = 1e4,
        collection_sample_size  = 1e4,
        # Removals parameters
        ff_frac = 0.2, # % of feed forward neurons to prune
        attn_frac   = 0.00, # % of attention neurons to prune
        focus   = focus_repo, # the “reference” dataset
        cripple = "physics", # the “unlearned” dataset
        additional_datasets = tuple(), # any extra datasets to evaluate on
        recalculate_activations = False, # iterative vs non-iterative
        n_steps = 1,
        save    = True,
        save_subdirectory   = "/home/ubuntu/tetra/taker/examples/neuron-mapping",
        evaluate_after_pruning  = False
    )

    # Parse CLI for arguments
    # c, args = cli_parser(c)

    #list of repos to cripple
    ff_frac_to_prune = 0.01
    model_size = c.model_repo.split('-')[-1] if model_size is None else model_size

    # Run the iterated pruning for each cripple repo, for a range of ff_frac pruned
    shared_pruning_data = {}
    c.ff_frac = ff_frac_to_prune
    for repo in cripple_repos:
        c.cripple = repo
        print("running iteration for ", c.cripple, " vs ", c.focus, "with ff_frac: ", ff_frac)
        with torch.no_grad():
            model, history = run_pruning(c)
    # ratios = compare_pruned_ff_criteria(cripple_repos, model_size)
    # shared_pruning_data[ff_frac] = ratios
    return {}

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


shared_pruning_data = get_shared_pruning_data(
    model_repo="nickypro/vit-cifar100-random-init",
    cripple_repos=datasets,
    focus_repo="cifar20-split")
print(shared_pruning_data)

# shared_pruning_data = get_shared_pruning_data("nickypro/tinyllama-15M")
# print(shared_pruning_data)
