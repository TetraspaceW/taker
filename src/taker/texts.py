"""
Some commands used for loading datasets used in my research.
That is, the 'codeparrot-clean' and 'the pile' datasets.
"""

import os
import json
import argparse
from datasets import load_dataset, concatenate_datasets

from .data_classes import EvalConfig
from .model import Model

# For each of these, we add a "test" argument:
#     If test == 0: use the "train" split
#     If test > 0 and there is a "test" split: return the "test" split
#     Else, return the train split with a skip of approx "test" tokens

# Hard load the most common tokens from the datasets from previous runs.
# pylint: disable=line-too-long
opt_most_common_code_tokens = [' ', '\n', '.', '_', ',', '#', '(', ' =', ' import', 'from', ' the', ':', ')', '\n\n', 'import', " '", '/', '-', '):', '\t', "',", ' "', ' self', '=', ' of', "'", '__', ' (', 'self', ' in', ' License', '</s>', ' is', '0', ' for', ' to', 's', '1', '2', ' a', ' as', '\r', ' -', ' and', ' def', ' #', 'x', '()', "('", '\\']
opt_most_common_pile_tokens = ['\n', '.', ',', ' the', ' ', ' of', ' to', ' and', ' a', ' in', '-', '</s>', ' is', ':', ' for', ' (', ' on', ')', ' with', ' that', ' I', '/', '�', ' as', ' by', ' was', ' an', 's', '�', 'The', ' are', ' The', ' it', ' have', ' from', ' this', ' be', ' at', ' you', '1', ' or', ' "', 'I', "'s", ' has', ' can', '"', ' -', '2', '?']

# Load the JSON data
def script_path(filename):
    __script_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(__script_path, filename)


json_file_path = script_path('data/llama_most_common_tokens.json')
with open(json_file_path, 'r') as file:
    llama_most_common_tokens = json.load(file)
most_common_pile_tokens          = llama_most_common_tokens["all"]["skip50"]["tokens_str"]
most_common_pile_codeless_tokens = llama_most_common_tokens["only_text"]["skip50"]["tokens_str"]
most_common_code_tokens          = llama_most_common_tokens["only_code"]["skip50"]["tokens_str"]

class DatasetFilters:
    @staticmethod
    def filter_codeless(_dataset):
        code_labels = set(["Github"])
        def filter_codeless_example(example):
            return str(example["meta"]["pile_set_name"]) not in code_labels
        rocket_dataset = _dataset.filter(filter_codeless_example)
        return rocket_dataset

    @staticmethod
    def filter_pile_general(_dataset, label):
        def filter_pile_example(example):
            return str(example["meta"]["pile_set_name"]) == label
        pile_filtered_dataset = _dataset.filter(filter_pile_example)
        return pile_filtered_dataset

    @staticmethod
    def filter_civil(_dataset):
        def filter_toxicity_example(example):
            return example["toxicity"] <= 0.2
        low_toxicity_dataset = _dataset.filter(filter_toxicity_example)
        return low_toxicity_dataset

    @staticmethod
    def filter_toxic(_dataset):
        def filter_toxicity_example(example):
            return example["toxicity"] >= 0.8
        toxic_dataset = _dataset.filter(filter_toxicity_example)
        return toxic_dataset

    @staticmethod
    def filter_birds(_dataset):
        with open(script_path("data/imagenet_birds.json"), "r") as file:
            bird_json = json.load(file)
        bird_ids = set(bird_json["id2label"].keys())
        def filter_birds_example(example):
            return str(example["label"]) in bird_ids
        bird_dataset = _dataset.filter(filter_birds_example)
        return bird_dataset

    @staticmethod
    def filter_birdless(_dataset):
        with open(script_path("data/imagenet_birds.json"), "r") as file:
            bird_json = json.load(file)
        bird_ids = set(bird_json["id2label"].keys())
        def filter_birds_out_example(example):
            return str(example["label"]) not in bird_ids
        bird_dataset = _dataset.filter(filter_birds_out_example)
        return bird_dataset

    @staticmethod
    def filter_mushroom(_dataset):
        mushroom_ids = set([ "52" ])
        def filter_mushroom_example(example):
            return str(example["fine_label"]) in mushroom_ids
        mushroom_dataset = _dataset.filter(filter_mushroom_example)
        return mushroom_dataset

    @staticmethod
    def filter_mushroomless(_dataset):
        mushroom_ids = set([ "52" ])
        def filter_mushroom_out_example(example):
            return str(example["fine_label"]) not in mushroom_ids
        mushroomless_dataset = _dataset.filter(filter_mushroom_out_example)
        return mushroomless_dataset

    @staticmethod
    def filter_rocket(_dataset):
        rocket_ids = set([ "69" ])
        def filter_rocket_example(example):
            return str(example["fine_label"]) in rocket_ids
        rocket_dataset = _dataset.filter(filter_rocket_example)
        return rocket_dataset

    @staticmethod
    def filter_rocketless(_dataset):
        rocket_ids = set([ "69" ])
        def filter_rocket_out_example(example):
            return str(example["fine_label"]) not in rocket_ids
        rocketless_dataset = _dataset.filter(filter_rocket_out_example)
        return rocketless_dataset
    
    @staticmethod
    def filter_aquatic_mammals(_dataset):
        aquatic_mammals_id = "0"
        def filter_aquatic_mammals_example(example):
            return str(example["coarse_label"]) == aquatic_mammals_id
        aquatic_mammals_dataset = _dataset.filter(filter_aquatic_mammals_example)
        return aquatic_mammals_dataset
    
    @staticmethod
    def filter_fish(_dataset):
        fish_id = "1"
        def filter_fish_example(example):
            return str(example["coarse_label"]) == fish_id
        fish_dataset = _dataset.filter(filter_fish_example)
        return fish_dataset
    
    @staticmethod
    def filter_flowers(_dataset):
        flowers_id = "2"
        def filter_flowers_example(example):
            return str(example["coarse_label"]) == flowers_id
        flowers_dataset = _dataset.filter(filter_flowers_example)
        return flowers_dataset
    
    @staticmethod
    def filter_food_containers(_dataset):
        food_containers_id = "3"
        def filter_food_containers_example(example):
            return str(example["coarse_label"]) == food_containers_id
        food_containers_dataset = _dataset.filter(filter_food_containers_example)
        return food_containers_dataset
    
    @staticmethod
    def filter_fruit_and_vegetables(_dataset):
        fruit_and_vegetables_id = "4"
        def filter_fruit_and_vegetables_example(example):
            return str(example["coarse_label"]) == fruit_and_vegetables_id
        fruit_and_vegetables_dataset = _dataset.filter(filter_fruit_and_vegetables_example)
        return fruit_and_vegetables_dataset
    
    @staticmethod
    def filter_household_electrical_devices(_dataset):
        household_electrical_devices_id = "5"
        def filter_household_electrical_devices_example(example):
            return str(example["coarse_label"]) == household_electrical_devices_id
        household_electrical_devices_dataset = _dataset.filter(filter_household_electrical_devices_example)
        return household_electrical_devices_dataset
    
    @staticmethod
    def filter_household_furniture(_dataset):
        household_furniture_id = "6"
        def filter_household_furniture_example(example):
            return str(example["coarse_label"]) == household_furniture_id
        household_furniture_dataset = _dataset.filter(filter_household_furniture_example)
        return household_furniture_dataset
    
    @staticmethod
    def filter_insects(_dataset):
        insects_id = "7"
        def filter_insects_example(example):
            return str(example["coarse_label"]) == insects_id
        insects_dataset = _dataset.filter(filter_insects_example)
        return insects_dataset
    
    @staticmethod
    def filter_large_carnivores(_dataset):
        large_carnivores_id = "8"
        def filter_large_carnivores_example(example):
            return str(example["coarse_label"]) == large_carnivores_id
        large_carnivores_dataset = _dataset.filter(filter_large_carnivores_example)
        return large_carnivores_dataset

    @staticmethod
    def filter_large_manmade_outdoor_things(_dataset):
        large_manmade_outdoor_things_id = "9"
        def filter_large_manmade_outdoor_things_example(example):
            return str(example["coarse_label"]) == large_manmade_outdoor_things_id
        large_manmade_outdoor_things_dataset = _dataset.filter(filter_large_manmade_outdoor_things_example)
        return large_manmade_outdoor_things_dataset
    
    @staticmethod
    def filter_large_natural_outdoor_scenes(_dataset):
        large_natural_outdoor_scenes_id = "10"
        def filter_large_natural_outdoor_scenes_example(example):
            return str(example["coarse_label"]) == large_natural_outdoor_scenes_id
        large_natural_outdoor_scenes_dataset = _dataset.filter(filter_large_natural_outdoor_scenes_example)
        return large_natural_outdoor_scenes_dataset
    
    @staticmethod
    def filter_large_omnivores_and_herbivores(_dataset):
        large_omnivores_and_herbivores_id = "11"
        def filter_large_omnivores_and_herbivores_example(example):
            return str(example["coarse_label"]) == large_omnivores_and_herbivores_id
        large_omnivores_and_herbivores_dataset = _dataset.filter(filter_large_omnivores_and_herbivores_example)
        return large_omnivores_and_herbivores_dataset
    
    @staticmethod
    def filter_medium_sized_mammals(_dataset):
        medium_sized_mammals_id = "12"
        def filter_medium_sized_mammals_example(example):
            return str(example["coarse_label"]) == medium_sized_mammals_id
        medium_sized_mammals_dataset = _dataset.filter(filter_medium_sized_mammals_example)
        return medium_sized_mammals_dataset
    
    @staticmethod
    def filter_non_insect_invertebrates(_dataset):
        non_insect_invertebrates_id = "13"
        def filter_non_insect_invertebrates_example(example):
            return str(example["coarse_label"]) == non_insect_invertebrates_id
        non_insect_invertebrates_dataset = _dataset.filter(filter_non_insect_invertebrates_example)
        return non_insect_invertebrates_dataset
    
    @staticmethod
    def filter_people(_dataset):
        people_id = "14"
        def filter_people_example(example):
            return str(example["coarse_label"]) == people_id
        people_dataset = _dataset.filter(filter_people_example)
        return people_dataset
    
    @staticmethod
    def filter_reptiles(_dataset):
        reptiles_id = "15"
        def filter_reptiles_example(example):
            return str(example["coarse_label"]) == reptiles_id
        reptiles_dataset = _dataset.filter(filter_reptiles_example)
        return reptiles_dataset
    
    @staticmethod
    def filter_small_mammals(_dataset):
        small_mammals_id = "16"
        def filter_small_mammals_example(example):
            return str(example["coarse_label"]) == small_mammals_id
        small_mammals_dataset = _dataset.filter(filter_small_mammals_example)
        return small_mammals_dataset

    @staticmethod
    def filter_trees(_dataset):
        trees_id = "17"
        def filter_trees_example(example):
            return str(example["coarse_label"]) == trees_id
        trees_dataset = _dataset.filter(filter_trees_example)
        return trees_dataset

    @staticmethod
    def filter_veh1(_dataset):
        veh1_id = "18"
        def filter_veh1_example(example):
            return str(example["coarse_label"]) == veh1_id
        veh1_dataset = _dataset.filter(filter_veh1_example)
        return veh1_dataset

    @staticmethod
    def filter_veh2(_dataset):
        rocket_ids = set([ "19" ])
        def filter_rocket_example(example):
            return str(example["coarse_label"]) in rocket_ids
        rocket_dataset = _dataset.filter(filter_rocket_example)
        return rocket_dataset
    
    @staticmethod
    def filter_pile_FreeLaw(_dataset):
        def filter_pile_FreeLaw_example(example):
            return example["meta"]["pile_set_name"] == "FreeLaw"
        pile_FreeLaw_dataset = _dataset.filter(filter_pile_FreeLaw_example)
        return pile_FreeLaw_dataset
    
    @staticmethod
    def filter_pile_PubMed_Abstracts(_dataset):
        def filter_pile_PubMed_Abstracts_example(example):
            return example["meta"]["pile_set_name"] == "PubMed Abstracts"
        pile_PubMed_Abstracts_dataset = _dataset.filter(filter_pile_PubMed_Abstracts_example)
        return pile_PubMed_Abstracts_dataset
    
    @staticmethod
    def filter_pile_PubMed_Central(_dataset):
        def filter_pile_PubMed_Central_example(example):
            return example["meta"]["pile_set_name"] == "PubMed Central"
        pile_PubMed_Central_dataset = _dataset.filter(filter_pile_PubMed_Central_example)
        return pile_PubMed_Central_dataset
    
    @staticmethod
    def filter_pile_NIH_ExPorter(_dataset):
        def filter_pile_NIH_ExPorter_example(example):
            return example["meta"]["pile_set_name"] == "NIH ExPorter"
        pile_NIH_ExPorter_dataset = _dataset.filter(filter_pile_NIH_ExPorter_example)
        return pile_NIH_ExPorter_dataset
    
    @staticmethod
    def filter_pile_Enron_Emails(_dataset):
        def filter_pile_Enron_Emails_example(example):
            return example["meta"]["pile_set_name"] == "Enron Emails"
        pile_Enron_Emails_dataset = _dataset.filter(filter_pile_Enron_Emails_example)
        return pile_Enron_Emails_dataset
        
    @staticmethod
    def filter_pile_Github(_dataset):
        def filter_pile_Github_example(example):
            return example["meta"]["pile_set_name"] == "Github"
        pile_Github_dataset = _dataset.filter(filter_pile_Github_example)
        return pile_Github_dataset
        
    @staticmethod
    def filter_pile_StackExchange(_dataset):
        def filter_pile_StackExchange_example(example):
            return example["meta"]["pile_set_name"] == "StackExchange"
        pile_StackExchange_dataset = _dataset.filter(filter_pile_StackExchange_example)
        return pile_StackExchange_dataset
    
    @staticmethod
    def filter_pile_HackerNews(_dataset):
        def filter_pile_HackerNews_example(example):
            return example["meta"]["pile_set_name"] == "HackerNews"
        pile_HackerNews_dataset = _dataset.filter(filter_pile_HackerNews_example)
        return pile_HackerNews_dataset
        
    @staticmethod
    def filter_pile_ArXiv(_dataset):
        def filter_pile_ArXiv_example(example):
            return example["meta"]["pile_set_name"] == "ArXiv"
        pile_ArXiv_dataset = _dataset.filter(filter_pile_ArXiv_example)
        return pile_ArXiv_dataset
        
    @staticmethod
    def filter_pile_Wikipedia(_dataset):
        def filter_pile_Wikipedia_example(example):
            return example["meta"]["pile_set_name"] == "Wikipedia (en)"
        pile_Wikipedia_dataset = _dataset.filter(filter_pile_Wikipedia_example)
        return pile_Wikipedia_dataset
    
    @staticmethod
    def filter_pile_Ubuntu_IRC(_dataset):
        def filter_pile_Ubuntu_IRC_example(example):
            return example["meta"]["pile_set_name"] == "Ubuntu IRC"
        pile_Ubuntu_IRC_dataset = _dataset.filter(filter_pile_Ubuntu_IRC_example)
        return pile_Ubuntu_IRC_dataset
    
    @staticmethod
    def filter_pile_USPTO_Backgrounds(_dataset):
        def filter_pile_USPTO_Backgrounds_example(example):
            return example["meta"]["pile_set_name"] == "USPTO Backgrounds"
        pile_USPTO_Backgrounds_dataset = _dataset.filter(filter_pile_USPTO_Backgrounds_example)
        return pile_USPTO_Backgrounds_dataset
    
    @staticmethod
    def filter_pile_PhilPapers(_dataset):
        def filter_pile_PhilPapers_example(example):
            return example["meta"]["pile_set_name"] == "PhilPapers"
        pile_PhilPapers_dataset = _dataset.filter(filter_pile_PhilPapers_example)
        return pile_PhilPapers_dataset
    
    @staticmethod
    def filter_pile_EuroParl(_dataset):
        def filter_pile_EuroParl_example(example):
            return example["meta"]["pile_set_name"] == "EuroParl"
        pile_EuroParl_dataset = _dataset.filter(filter_pile_EuroParl_example)
        return pile_EuroParl_dataset
    
    @staticmethod
    def filter_pile_Gutenberg(_dataset):
        def filter_pile_Gutenberg_example(example):
            return example["meta"]["pile_set_name"] == "Gutenberg (PG-19)"
        pile_Gutenberg_dataset = _dataset.filter(filter_pile_Gutenberg_example)
        return pile_Gutenberg_dataset
    
    @staticmethod
    def filter_cifar(id: str):
        def filter_example(example):
            return str(example["coarse_label"]) == str(id)
        def filter_dataset(_dataset):
            return _dataset.filter(filter_example)
        return filter_dataset
        
def get_cifar_dataset_configs():
    cifar20_datasets = ["aquatic_mammals", "fish", "flowers", "food_containers", "fruit_and_vegetables", "household_electrical_devices", "household_furniture", "insects", "large_carnivores", "large_outdoor", "large_omnivores_and_herbivores", "medium_mammals", "non_insect_invertebrates", "people", "reptiles", "small_mammals", "trees", "veh1", "veh2"]
    return [EvalConfig(f"cifar20-{dataset}",
                       dataset_repo = "cifar100",
                       dataset_type = "image-classification",
                       dataset_split = ["train", "test"],
                       is_train_mode = True,
                       dataset_image_key = "img",
                       streaming = False,
                       dataset_image_label_key = "fine_label",
                       dataset_filter=DatasetFilters.filter_cifar(count),
                       ) for count, dataset in enumerate(cifar20_datasets)]

def infer_dataset_config(dataset_name:str, dataset_subset:str=None):
    eval_configs = [
        EvalConfig("pile_codeless",
            dataset_repo = "monology/pile-uncopyrighted",
            skip_token_strings = most_common_pile_codeless_tokens,
            dataset_filter = DatasetFilters.filter_codeless,
        ),
        EvalConfig("pile_freelaw",
            dataset_repo = "monology/pile-uncopyrighted",
            skip_token_strings = most_common_pile_codeless_tokens,
            dataset_filter = lambda __dataset : DatasetFilters.filter_pile_general(__dataset, "FreeLaw"),
        ),
        EvalConfig("pile",
            dataset_repo = "monology/pile-uncopyrighted",
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("code",
            dataset_repo           = "codeparrot/github-code-clean",
            dataset_subset         = "all-all",
            dataset_text_key       = "code",
            dataset_has_test_split = False,
            skip_token_strings = most_common_code_tokens,
        ),
        EvalConfig("python",
            dataset_repo           = "codeparrot/github-code-clean",
            dataset_subset         = "Python-all",
            dataset_text_key       = "code",
            dataset_has_test_split = False,
            skip_token_strings = most_common_code_tokens,
        ),
        EvalConfig("stories",
            dataset_repo = "roneneldan/TinyStories",
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("civil",
            dataset_repo = "civil_comments",
            dataset_filter = DatasetFilters.filter_civil,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("toxic",
            dataset_repo = "civil_comments",
            dataset_filter = DatasetFilters.filter_toxic,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("wiki",
            dataset_repo = "wikitext",
            dataset_subset = "wikitext-103-v1",
            sample_size = int(1e6),
            skip_token_strings = opt_most_common_pile_tokens,
        ),
        EvalConfig("toxicity",
            dataset_repo = None,
            dataset_type = "generation",
            generated_text_prompt = "I don’t care if this is controversial",
            #generated_text_prompt = " I swear, I just don’t know if",
            generated_text_length = 200,
            generated_text_include_prompt = True,
            generated_text_num_samples = 1000,
            generated_text_temperature = 1.0,
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("mmlu",
            dataset_repo = "tasksource/mmlu",
            dataset_type = "mmlu",
            dataset_subset = "all", # Overwritten if use "mmlu:subject_name"
            skip_token_strings = most_common_pile_tokens,
        ),
        EvalConfig("imagenet-1k",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
        ),
        EvalConfig("imagenet-1k-birds",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
            dataset_filter=DatasetFilters.filter_birds,
        ),
        EvalConfig("imagenet-1k-birdless",
            dataset_split = "validation",
            dataset_repo = "imagenet-1k",
            dataset_type = "image-classification",
            dataset_filter=DatasetFilters.filter_birdless,
        ),
        EvalConfig("cifar100",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_image_key = "img",
            num_texts_to_skip = 1,
            dataset_image_label_key = "fine_label",
        ),
        EvalConfig("cifar100-mushroom",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_split = ["train", "test"],
            is_train_mode = True,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_mushroom,
        ),
        EvalConfig("cifar100-mushroomless",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            streaming = False,
            dataset_split = "test",
            is_train_mode = False,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_mushroomless,
        ),
        EvalConfig("cifar100-mushroom-mia",
            dataset_repo = "cifar100",
            dataset_type = "image-membership-inference-attack",
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            mia_retain = "cifar100-mushroomless",
            mia_retain_split = "train",
            mia_forget = "cifar100-mushroom",
            mia_forget_split = "train",
            mia_test = "cifar100",
            mia_test_split = "test",
            dataset_filter=DatasetFilters.filter_mushroom,
        ),
        EvalConfig("cifar100-rocket",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = ["train", "test"],
            streaming = False,
            is_train_mode = True,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_rocket,
        ),
        EvalConfig("cifar100-rocketless",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = "test",
            streaming = False,
            is_train_mode = False,
            num_texts_to_skip = 1,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            dataset_filter=DatasetFilters.filter_rocketless,
        ),
        EvalConfig("cifar100-rocket-mia",
            dataset_repo = "cifar100",
            dataset_type = "image-membership-inference-attack",
            is_train_mode = True,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
            mia_retain = "cifar100-rocketless",
            mia_retain_split = "train",
            mia_forget = "cifar100-rocket",
            mia_forget_split = "train",
            mia_test = "cifar100",
            mia_test_split = "test",
            dataset_filter=DatasetFilters.filter_rocket,
        ),
        EvalConfig("cifar20",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
        ),
        EvalConfig("cifar20-split",
            dataset_repo = "cifar100",
            dataset_type = "image-classification",
            dataset_split = ["train", "test"],
            is_train_mode = True,
            streaming = False,
            dataset_image_key = "img",
            dataset_image_label_key = "fine_label",
        ),
        EvalConfig("bio",
            dataset_repo           = "camel-ai/biology",
            dataset_text_key       = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("emotion",
            dataset_repo = "dair-ai/emotion",
            dataset_type = "text-classification",
            dataset_text_key = "text",
            dataset_text_label_key = "label",
            dataset_has_test_split = True,
        ),
        EvalConfig("biology",
            dataset_repo           = "camel-ai/biology",
            dataset_text_key       = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("physics",
            dataset_repo = "camel-ai/physics",
            dataset_text_key = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("chemistry",
            dataset_repo = "camel-ai/chemistry",
            dataset_text_key = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("math",
            dataset_repo = "camel-ai/math",
            dataset_text_key = "message_2",
            dataset_has_test_split = False,
        ),
        EvalConfig("poems",
            dataset_repo = "sadFaceEmoji/english-poems",
            dataset_text_key = "poem",
            dataset_has_test_split = False,
        ),
        EvalConfig("pile_FreeLaw",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_FreeLaw,
        ),
        EvalConfig("pile_PubMed_Abstracts",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_PubMed_Abstracts,
        ),
        EvalConfig("pile_PubMed_Central",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_PubMed_Central,
        ),
        EvalConfig("pile_NIH_ExPorter",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_NIH_ExPorter,
        ),
        EvalConfig("pile_Enron_Emails",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_Enron_Emails,
        ),
        EvalConfig("pile_Github",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_Github,
        ),
        EvalConfig("pile_StackExchange",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_StackExchange,
        ),
        EvalConfig("pile_HackerNews",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_HackerNews,
        ),
        EvalConfig("pile_ArXiv",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_ArXiv,
        ),
        EvalConfig("pile_Wikipedia",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_Wikipedia,
        ),
        EvalConfig("pile_Ubuntu_IRC",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_Ubuntu_IRC,
        ),
        EvalConfig("pile_USPTO_Backgrounds",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_USPTO_Backgrounds,
        ),
        EvalConfig("pile_PhilPapers",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_PhilPapers,
        ),
        EvalConfig("pile_EuroParl",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_EuroParl,
        ),
        EvalConfig("pile_Gutenberg",
            dataset_repo = "monology/pile-uncopyrighted",
            dataset_text_key = "text",
            dataset_filter = DatasetFilters.filter_pile_Gutenberg,
        ),
    ] + get_cifar_dataset_configs()

    # Convert into searchable dict
    labeled_eval_configs = dict([(c.dataset_name, c) for c in eval_configs])

    # Search the dict for config
    if dataset_name in labeled_eval_configs:
        eval_config = labeled_eval_configs[dataset_name]
    else:
        eval_config = EvalConfig(dataset_name)

    # Add subset data
    if dataset_subset is not None:
        eval_config.dataset_subset = dataset_subset

    # Add loading bar label if there is none
    if eval_config.loading_bar_desc is None or eval_config.loading_bar_desc == "":
        eval_config.loading_bar_desc = "%6s" % eval_config.dataset_name

    return eval_config

def prepare_dataset(eval_config: EvalConfig):
    """ Returns iterable dataset object. """

    # check if it has test split, or only a train split
    split = eval_config.dataset_split
    if split is None:
        split = "test" if eval_config.dataset_has_test_split else "train"

    # Load the dataset
    _dataset = load_dataset(
        eval_config.dataset_repo,
        eval_config.dataset_subset,
        streaming=eval_config.streaming,
    )


    # Post-split processing
    if isinstance(split, list) or isinstance(split, tuple):
        __d = [_dataset[s] for s in split]
        _dataset = concatenate_datasets(__d)

    else:
        _dataset = _dataset[split]

    # Apply filter if relevant
    if eval_config.dataset_filter is not None:
        _dataset = eval_config.dataset_filter(_dataset)
    # Skip n texts if relevant
    if eval_config.num_texts_to_skip >= 1:
        print(f"skipping {eval_config.num_texts_to_skip} texts in {eval_config.dataset_name}")

        # Skip only works for DatasetIterable. Kinda annoying ngl
        if hasattr(_dataset, "skip"):
            _dataset = _dataset.skip(eval_config.num_texts_to_skip) # Conservative skip limit
        else:
            indices = list(range(eval_config.num_texts_to_skip, len(_dataset)))
            _dataset = _dataset.select(indices)

    # Skip tokens is no split
    if split == "train" and not eval_config.is_train_mode:
        skip_n = int(eval_config.num_tokens_to_skip//100)
        print( "Warning: 'pile_deduped' has no 'test' split.",
              f"Using 'train' split and skipping {skip_n} texts instead.")
        _dataset = _dataset.skip(skip_n) # Conservative skip limit

    return _dataset

def prepare(dataset_name):
    eval_config = infer_dataset_config(dataset_name)
    eval_config.dataset_split = "train"
    _dataset = prepare_dataset(eval_config)
    return _dataset, eval_config.dataset_text_key, eval_config.skip_token_strings
