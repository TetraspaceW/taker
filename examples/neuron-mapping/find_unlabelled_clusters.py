from taker.data_classes import PruningConfig
from taker.model import Model
from taker.prune import prune_and_evaluate, run_pruning

c = PruningConfig(
    wandb_project="testing",  # repo to push results to
    model_repo="Ahmed9275/Vit-Cifar100",
    token_limit=1000,  # trim the input to this max length
    run_pre_test=False,  # evaluate the unpruned model
    eval_sample_size=1e3,
    collection_sample_size=1e3,
    # Removals parameters
    ff_frac=0.01,  # % of feed forward neurons to prune
    attn_frac=0.00,  # % of attention neurons to prune
    focus="cifar20-split",  # the “reference” dataset
    cripple="cifar20-aquatic_mammals",  # the “unlearned” dataset
    additional_datasets=tuple(),  # any extra datasets to evaluate on
    recalculate_activations=True,  # iterative vs non-iterative
    dtype="fp32",
    n_steps=10,
)

model = Model(
    c.model_repo, limit=c.token_limit, dtype=c.dtype, use_accelerator=c.use_accelerator
)

# get the activations
