import asyncio
import dataclasses
from dataclasses import dataclass, field
import datetime
from enum import StrEnum
import json
import logging
import math
import os
import random
import re
import statistics
import sys
from typing import Any, Dict, Union

import graphviz
import httpx
import rdflib
import tiktoken


# Set-up and Config

# Check for existence of all directories:

# Set parameters for ontology and deal with input edge cases
chosen_model = "gpt-4-0613"
model_cost_output = 0.002
model_cost_input = 0.0015

def set_dirs():
    if not os.path.exists("credentials"):
        os.makedirs("credentials")
    if not os.path.exists("ontologies"):
        os.makedirs("ontologies")
    if not os.path.exists("query_cache"):
        os.makedirs("query_cache")


# Initialize config with default values
def set_config():
    if not os.path.exists("config.json"):
        config_initial = {
            "Caching preferences": {"Prompt caching": True, "Sample caching": False},
            "Thresholds": {
                "Cost threshold (in dollar)": 30,
                "Concept threshold": 3000,
                "Time threshold (minutes)": 300,
            },
            "Prompt for parameters": {
                "Exploration depth": True,
                "Frequency threshold": True,
                "Outdegree": False,
                "temperature (sampling)": False,
                "top_p (sampling)": False,
                "n (sampling)": False,
            },
            "Default values for parameters": {
                "Exploration depth": 1,
                "Frequency threshold": 10,
                "Outdegree": 100000,
                "temperature (sampling)": 2.0,
                "top_p (sampling)": 0.99,
                "n (sampling)": 100,
            },
        }
        with open("config.json", "w", encoding="utf8") as file:
            json.dump(config_initial, file, indent=4)
        print(
            "Initialized config with default values. You can change your preferences after providing an API key."
        )


# Set API key

# Set organization ID


def set_organization_id():
    if not os.path.exists("credentials/openai_organization_id.txt"):
        new_org_id = input(
            "Optional: Provide your organization ID (it will be stored under credentials/openai_organization_id.txt): "
        )
        if new_org_id:
            with open(
                "credentials/openai_organization_id.txt", "w", encoding="utf8"
            ) as id_file:
                id_file.write(new_org_id)
        else:
            print("No organization ID provided. Continuing only with API key.")
    else:
        new_org_id = input(
            "Provide your new organization ID for the openAI API (it will be stored under credentials/openai_organization_id.txt). No input will fall back on the existing organization ID (if any). To delete your existing ID type 'DEL': "
        )
        if new_org_id == "DEL":
            if os.path.exists("credentials/openai_organization_id.txt"):
                os.remove("credentials/openai_organization_id.txt")
            print("The organization ID has been deleted.")
        elif new_org_id:
            with open("credentials/openai_organization_id.txt", "w", encoding="utf8") as id_file:
                id_file.write(new_org_id)
            print(f"Set new organization ID: {new_org_id}")
        else:
            print("No new organization ID provided. Falling back on existing ID.")


def set_api_key():
    if not os.path.exists("credentials/openai_key.txt"):
        new_api_key = input(
            "In order to use this program you need to provide an API key to the openAI API (it will be stored under credentials/openai_key.txt): "
        )
        while not new_api_key:
            new_api_key = input(
                "In order to use this program you need to provide an API key to the openAI API (it will be stored under credentials/openai_key.txt): "
            )
        with open("credentials/openai_key.txt", "w", encoding="utf8") as key_file:
            key_file.write(new_api_key)
        print(f"Set API key: {new_api_key}")
    else:
        new_api_key = input(
            "Provide your new API key to the openAI API (it will be stored under credentials/openai_key.txt). No input will fall back on the existing API key: "
        )
        if new_api_key:
            with open("credentials/openai_key.txt", "w", encoding="utf8") as key_file:
                key_file.write(new_api_key)
            print(f"Set new API key: {new_api_key}")
        else:
            print("No new API key provided. Falling back on existing key.")


# Set caching preferences


def set_caching():
    with open("config.json", "r", encoding="utf8") as config_file:
        config = json.load(config_file)
        if config["Caching preferences"]["Prompt caching"] == True:
            change_prompt_caching = input(
                "Prompt caching is activated. Do you want to deactivate it? Answer Y/N: "
            )
            if change_prompt_caching in {"Yes", "yes", "Y", "y"}:
                config["Caching preferences"]["Prompt caching"] = False
                print("Prompt caching is now deactivated")
        else:
            change_prompt_caching = input(
                "Prompt caching is deactivated. Do you want to activate it? Answer Y/N: "
            )
            if change_prompt_caching in {"Yes", "yes", "Y", "y"}:
                config["Caching preferences"]["Prompt caching"] = True
                print("Prompt caching is now activated")
        if config["Caching preferences"]["Sample caching"] == True:
            change_prompt_caching = input(
                "Sample caching is activated. Do you want to deactivate it? Answer Y/N: "
            )
            if change_prompt_caching in {"Yes", "yes", "Y", "y"}:
                config["Caching preferences"]["Sample caching"] = False
                print("Sample caching is now deactivated")
        else:
            change_prompt_caching = input(
                "Sample caching is deactivated. Do you want to activate it? Answer Y/N: "
            )
            if change_prompt_caching in {"Yes", "yes", "Y", "y"}:
                config["Caching preferences"]["Sample caching"] = True
                print("Sample caching is now activated")
    with open("config.json", "w", encoding="utf8") as updated_config_file:
        json.dump(config, updated_config_file, indent=4)


# Set threshold preferences


def set_thresholds():
    with open("config.json", "r", encoding="utf8") as config_file:
        config = json.load(config_file)

        change_cost_threshold = input(
            f'Cost threshold is set to {config["Thresholds"]["Cost threshold (in dollar)"]}$. Do you want to change it? Answer with Y/N: '
        )
        if change_cost_threshold in {"Yes", "yes", "Y", "y"}:
            new_cost_threshold = input(
                "Enter a new value for cost threshold (only positive integers will be accepted): "
            )
            try:
                new_cost_threshold = int(new_cost_threshold)
                if 0 < new_cost_threshold:
                    config["Thresholds"][
                        "Cost threshold (in dollar)"
                    ] = new_cost_threshold
                    print(f"Cost threshold is now set to {new_cost_threshold}$.")
                else:
                    print(
                        "No positive integer provided. Falling back on previous cost threshold."
                    )
            except ValueError:
                print("No integer provided. Falling back on previous cost threshold.")

        change_concept_threshold = input(
            f'Concept threshold is set to {config["Thresholds"]["Concept threshold"]}. Do you want to change it? Answer with Y/N: '
        )
        if change_concept_threshold in {"Yes", "yes", "Y", "y"}:
            new_concept_threshold = input(
                "Enter a new value for concept threshold (only positive integers will be accepted): "
            )
            try:
                new_concept_threshold = int(new_concept_threshold)
                if 0 < new_concept_threshold:
                    config["Thresholds"]["Concept threshold"] = new_concept_threshold
                    print(f"Concept threshold is now set to {new_concept_threshold}.")
                else:
                    print(
                        "No positive integer provided. Falling back on previous concept threshold."
                    )
            except ValueError:
                print(
                    "No integer provided. Falling back on previous concept threshold."
                )

        change_time_threshold = input(
            f'Time threshold is set to {config["Thresholds"]["Time threshold (minutes)"]} minutes. Do you want to change it? Answer with Y/N: '
        )
        if change_time_threshold in {"Yes", "yes", "Y", "y"}:
            new_time_threshold = input(
                "Enter a new value for time threshold (only positive integers will be accepted): "
            )
            try:
                new_time_threshold = int(new_time_threshold)
                if 0 < new_time_threshold:
                    config["Thresholds"][
                        "Time threshold (minutes)"
                    ] = new_time_threshold
                    print(f"Time threshold is now set to {new_time_threshold} minutes.")
                else:
                    print(
                        "No positive integer provided. Falling back on previous time threshold."
                    )
            except ValueError:
                print("No integer provided. Falling back on previous time threshold.")
    with open("config.json", "w", encoding="utf8") as updated_config_file:
        json.dump(config, updated_config_file, indent=4)


# Activate/Deactivate prompting for additional parameters in the beginning


def set_parameters():
    with open("config.json", "r", encoding="utf8") as config_file:
        config = json.load(config_file)
        if config["Prompt for parameters"]["Outdegree"] == True:
            change_prompt_caching = input(
                "Parameter outdegree is activated. Do you want to deactivate it? Answer Y/N: "
            )
            if change_prompt_caching in {"Yes", "yes", "Y", "y"}:
                config["Prompt for parameters"]["Outdegree"] = False
                print("Parameter outdegree is now deactivated")
        else:
            change_prompt_caching = input(
                "Parameter outdegree is deactivated. Do you want to activate it? Answer Y/N: "
            )
            if change_prompt_caching in {"Yes", "yes", "Y", "y"}:
                config["Prompt for parameters"]["Outdegree"] = True
                print("Parameter outdegree is now activated")
    with open("config.json", "w", encoding="utf8") as updated_config_file:
        json.dump(config, updated_config_file, indent=4)


def change_preferences():
    enter_menu = input(
        "Do you want to change your preferences (set API key, organization ID, caching or termination thresholds)? Answer Y/N: "
    )
    if enter_menu in {"Yes", "yes", "Y", "y"}:
        menu_api = input(
            "Do you want to change your openAI credentials (API key or API organization ID)? Answer Y/N: "
        )
        if menu_api in {"Yes", "yes", "Y", "y"}:
            set_api_key()
            set_organization_id()
        menu_caching = input(
            "Do you want to change your caching preferences? Answer Y/N: "
        )
        if menu_caching in {"Yes", "yes", "Y", "y"}:
            set_caching()
        menu_threshold = input(
            "Do you want to change your threshold preferences? Answer Y/N: "
        )
        if menu_threshold in {"Yes", "yes", "Y", "y"}:
            set_thresholds()
        set_parameters()


def initialize_settings():
    set_dirs()
    if not os.path.exists("config.json"):
        set_config()
    if not os.path.exists("credentials/openai_key.txt"):
        set_api_key()
        set_organization_id()
    change_preferences()


initialize_settings()


with open("credentials/openai_key.txt", "r", encoding="utf8") as auth_key:
    auth_key = auth_key.read().strip()
api_key = auth_key

org = ""
if os.path.exists("credentials/openai_organization_id.txt"):
    with open("credentials/openai_organization_id.txt", "r", encoding="utf8") as org:
        org = org.read().strip()


# TOKEN ENCODING
def num_tokens_from_messages(messages, model="gpt-4-0613"):
    tokens_per_message = 0
    tokens_per_name = 0
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314", #it seems like it's no logner available -V
        "gpt-4-32k-0314", #it seems like it's no logner available -V
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301": #it seems like it's no logner available -V
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


# GLOBAL PARAMTERS
cost_threshold = None  # dollar
concept_threshold = None  # number of concepts
running_time_threshold = None  # in minutes
prompt_caching = None
sample_caching = None
with open("config.json", "r", encoding="utf8") as config_file:
    config = json.load(config_file)
    cost_threshold = config["Thresholds"]["Cost threshold (in dollar)"]
    concept_threshold = config["Thresholds"]["Concept threshold"]
    running_time_threshold = config["Thresholds"]["Time threshold (minutes)"]
    prompt_caching = config["Caching preferences"]["Prompt caching"]
    sample_caching = config["Caching preferences"]["Sample caching"]

bottom_concept = "No subconcept"
top_concept = "Thing"
initial_concept = ""
initial_concept_definition = ""


# INPUT PARAMETERS

# Initialize parameters
exploration_depth = None
frequency_threshold = None
max_outdegree = None
temperature_sampling = None
top_p_sampling = None
n_completions_sampling = None

# Prompt the user for the initial concept and depth of concept hierarchy
if len(sys.argv) == 8:
    initial_concept = sys.argv[1]
    exploration_depth = sys.argv[2]
    frequency_threshold = sys.argv[3]
    max_outdegree = sys.argv[4]
    temperature_sampling = sys.argv[5]
    top_p_sampling = sys.argv[6]
    n_completions_sampling = sys.argv[7]
elif len(sys.argv) == 5:
    initial_concept = sys.argv[1]
    exploration_depth = sys.argv[2]
    frequency_threshold = sys.argv[3]
    max_outdegree = sys.argv[4]
elif len(sys.argv) == 4:
    initial_concept = sys.argv[1]
    exploration_depth = sys.argv[2]
    frequency_threshold = sys.argv[3]
else:
    # If not provided, prompt the user for input
    with open("config.json", "r", encoding="utf8") as config_file:
        config = json.load(config_file)
    initial_concept = input("Enter the initial concept: ")
    if config["Prompt for parameters"]["Exploration depth"] is True:
        exploration_depth = input(
            "Enter the depth of subconcepts (range 1 to 100, defaults to 1): "
        )
    if config["Prompt for parameters"]["Frequency threshold"] is True:
        frequency_threshold = input(
            "Enter the number of times you want a token to be sampled to start a prompt for a list of subconcepts with it. Provide integer in range 1 to 100 (recommended: 5 to 20), defaults to 10: "
        )
    if config["Prompt for parameters"]["Outdegree"] is True:
        max_outdegree = input(
            "Enter the maximum number of immediate subconcepts explored for any concept (defaults to unlimited): "
        )
    if config["Prompt for parameters"]["temperature (sampling)"] is True:
        temperature_sampling = input(
            "Enter the temperature parameter for sampling of first token (float in range 0 to 2, defaults to 2.0): "
        )
    if config["Prompt for parameters"]["top_p (sampling)"] is True:
        top_p_sampling = input(
            "Enter the top_p parameter for sampling of first token (float in range 0 to 1, defaults to 0.99): "
        )
    if config["Prompt for parameters"]["n (sampling)"] is True:
        n_completions_sampling = input(
            "Enter the number of tokens you want to sample (integer in range 1 to 100, defaults to 100): "
        )


with open("config.json", "r", encoding="utf8") as config_file:
    config = json.load(config_file)
if not exploration_depth:
    exploration_depth = config["Default values for parameters"]["Exploration depth"]
elif not exploration_depth.isdigit() or int(exploration_depth) < 1:
    exploration_depth = config["Default values for parameters"]["Exploration depth"]
else:
    exploration_depth = int(exploration_depth)
if not max_outdegree:
    max_outdegree = config["Default values for parameters"]["Outdegree"]
elif not max_outdegree.isdigit() or int(max_outdegree) < 1:
    max_outdegree = config["Default values for parameters"]["Outdegree"]
else:
    max_outdegree = int(max_outdegree)
if not temperature_sampling:
    temperature_sampling = config["Default values for parameters"][
        "temperature (sampling)"
    ]
else:
    try:
        temperature_sampling = float(temperature_sampling)
        if 0.0 <= temperature_sampling <= 2.0:
            pass
        else:
            temperature_sampling = config["Default values for parameters"][
                "temperature (sampling)"
            ]
    except ValueError:
        temperature_sampling = config["Default values for parameters"][
            "temperature (sampling)"
        ]
if not top_p_sampling:
    top_p_sampling = config["Default values for parameters"]["top_p (sampling)"]
else:
    try:
        top_p_sampling = float(top_p_sampling)
        if 0.0 <= top_p_sampling <= 1.0:
            pass
        else:
            top_p_sampling = config["Default values for parameters"]["top_p (sampling)"]
    except ValueError:
        top_p_sampling = config["Default values for parameters"]["top_p (sampling)"]
if not n_completions_sampling:
    n_completions_int = config["Default values for parameters"]["n (sampling)"]
else:
    try:
        n_completions_int = int(n_completions_sampling)
        if 1 <= n_completions_int <= 100:
            pass
        else:
            n_completions_int = config["Default values for parameters"]["n (sampling)"]
    except ValueError:
        n_completions_int = config["Default values for parameters"]["n (sampling)"]
if not frequency_threshold:
    frequency_threshold = min(
        config["Default values for parameters"]["Frequency threshold"],
        config["Default values for parameters"]["n (sampling)"],
    )
else:
    try:
        frequency_threshold = int(frequency_threshold)
        if 1 <= frequency_threshold <= n_completions_int:
            pass
        else:
            frequency_threshold = config["Default values for parameters"][
                "n (sampling)"
            ]
    except ValueError:
        frequency_threshold = min(
            config["Default values for parameters"]["Frequency threshold"],
            config["Default values for parameters"]["n (sampling)"],
        )
if initial_concept in {top_concept, bottom_concept}:
    top_concept = "Top concept"
    bottom_concept = "Bottom concept"


if not initial_concept:
    print("No initial concept provided. Exiting.")
    sys.exit()

if max_outdegree < 100000:
    print(
        f'[{datetime.datetime.now().strftime("%H:%M:%S")}] Prompting {chosen_model} for subconcepts of {initial_concept} up to depth {exploration_depth}. Frequency threshold set to {frequency_threshold}. Maximum outdegree is set to {max_outdegree}.'
    )
else:
    print(
        f'[{datetime.datetime.now().strftime("%H:%M:%S")}] Prompting {chosen_model} for subconcepts of {initial_concept} up to depth {exploration_depth}. Frequency threshold set to {frequency_threshold}.'
    )

# NAMING SCHEME

# Make current branch available for log, metadata and ontology file naming

timestamp_start = datetime.datetime.now().replace(microsecond=0)
ontology_directory = (
    "ontologies/"
    + initial_concept.replace(" ", "_")
    + f"_{int(timestamp_start.timestamp())}"
    + f"_depth{exploration_depth}"
    + f"_ft{frequency_threshold}"
)
os.makedirs(ontology_directory, exist_ok=True)
filename = initial_concept.replace(" ", "_") + f"_{int(timestamp_start.timestamp())}"

filepath = os.path.join(
    ontology_directory,
    filename,
)


# METADATA AND LOGGING
metadata = {
    "model": chosen_model,
    "start": f"{str(timestamp_start)}",
    "end": "",
    "duration (seconds)": 0,
    "timestamp": int(timestamp_start.timestamp()),
    "seed concept": initial_concept,
    "prompts": 0,
    "duplicates": 0,
    "tokens (total)": 0,
    "tokens (responses)": 0,
    "tokens (requests)": 0,
    "parameters": {
        "temperature (sampling)": temperature_sampling,
        "top_p (sampling)": top_p_sampling,
        "number of completions (sampling)": n_completions_int,
        "frequency threshold for sampling": frequency_threshold,
        "exploration depth": exploration_depth,
    },
    "statistics": {
        "Concepts in ontology": 1,
        "Accepted after renaming": 0,
        "Dismissed terms": 0,
        "Dismissed as instance": 0,
        "Dismissed as partOf": 0,
        "Dismissed as no subconcept of seed": 0,
        "Dismissed as no subconcept of parent": 0,
        "Synonyms": 0,
        "Concepts without subconcept": 0,
        "Subsumptions (transitive reduction)": 0,
        "Subsumptions (found by KRIS)": 0,
        "Cost of ontology in dollar": 0,
        "Cost per concept in cent": 0,
        "Prompts": 0,
        "Prompts per concept": 0,
        "Tokens": 0,
        "Tokens per concept": 0,
        "Maximal depth": 0,
        "Maximal indegree": 0,
        "Maximal outdegree": 0,
        "Average degree": 0,
    },
}

# Create a custom logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# Create handlers
f_info_handler = logging.FileHandler("{}.log".format(filepath))
f_info_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
f_format = logging.Formatter("%(asctime)s %(levelname)s %(taskname)s %(message)s")
f_info_handler.setFormatter(f_format)
log.addHandler(f_info_handler)


prompt_log = logging.getLogger("prompts")
prompt_log.setLevel(logging.INFO)
f_prompt_handler = logging.FileHandler("{}_prompt.log".format(filepath))
f_prompt_handler.setFormatter(f_format)
prompt_log.addHandler(f_prompt_handler)

verification_log = logging.getLogger("verification")
verification_log.setLevel(logging.INFO)
f_verification_handler = logging.FileHandler("{}_verification.log".format(filepath))
f_verification_handler.setFormatter(f_format)
verification_log.addHandler(f_verification_handler)

old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
    record = old_factory(*args, **kwargs)
    try:
        t = asyncio.current_task()
        if t:
            record.taskname = t.get_name()
        else:
            record.taskname = "main"
    except Exception as e:
        record.taskname = "main"
    return record


logging.setLogRecordFactory(record_factory)

# Updating metadata and logs
metadata_lock = asyncio.Lock()


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def print_with_timestamp(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    t = asyncio.current_task()
    if t:
        print(f"[{timestamp}, {t.get_name()}] {message}")
    else:
        print(f"[{timestamp}, main] {message}")


async def update_metadata(response):
    async with metadata_lock:
        metadata["prompts"] += 1
        metadata["tokens (total)"] += response["usage"]["total_tokens"]
        metadata["tokens (responses)"] += response["usage"]["completion_tokens"]
        metadata["tokens (requests)"] += response["usage"]["prompt_tokens"]


async def update_metadata_from_cache(message_list, cached_response):
    async with metadata_lock:
        metadata["prompts"] += 1
        tokens_messages = num_tokens_from_messages(message_list)
        metadata["tokens (requests)"] += tokens_messages
        metadata["tokens (total)"] += tokens_messages
        tokens_response = len(
            tiktoken.encoding_for_model(chosen_model).encode(cached_response)
        )
        metadata["tokens (responses)"] += tokens_response
        metadata["tokens (total)"] += tokens_response


async def update_metadata_from_descriptions_cache(
    subconcepts, concept, definitions, new_definitions
):
    in_prompt = f"Give a brief description of every term on the list, considered as a subcategory of {concept}, without the use of examples, in the following form:\nList element 1: brief description of list element 1.\nList element 2: brief description of list element 2.\n..."
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages([initial_concept, concept], definitions)
    messages.append(
        {
            "role": "user",
            "content": f'List all of the most important subcategories of "{concept}" in a comma-separated format, skip explanations.',
        }
    )
    messages.append(
        {"role": "assistant", "content": ", ".join([s.name for s in subconcepts])}
    )
    messages.append(prompt_message)
    response = ""
    for k, v in new_definitions.items():
        response += f"{k}: {v}\n"
    async with metadata_lock:
        metadata["prompts"] += 1
        tokens_messages = num_tokens_from_messages(messages)
        metadata["tokens (requests)"] += tokens_messages
        metadata["tokens (total)"] += tokens_messages
        tokens_response = len(
            tiktoken.encoding_for_model(chosen_model).encode(response)
        )
        metadata["tokens (responses)"] += tokens_response
        metadata["tokens (total)"] += tokens_response


async def update_metadata_from_cache_sample(message_list, completion_number):
    async with metadata_lock:
        metadata["prompts"] += 1
        tokens_messages = num_tokens_from_messages(message_list)
        metadata["tokens (requests)"] += tokens_messages
        metadata["tokens (total)"] += tokens_messages
        metadata["tokens (responses)"] += completion_number
        metadata["tokens (total)"] += completion_number


async def print_and_log_prompt_response(messages, output):
    log_entry = "Prompt:\n"
    for message in messages:
        log_entry = log_entry + f'{message["content"]}\n'
    log_entry = log_entry + f"{output}"
    prompt_log.info(log_entry)


# GENERATING FILE OUTPUT


# JSON
def output_hierarchy_as_json(hierarchy, filepath):
    with open(f"{filepath}.json", "w", encoding="utf8") as file:
        json.dump(hierarchy, file, indent=4)


def hierarchy_to_rdf_graph(hierarchy: dict[str, dict[str, Any]]) -> rdflib.Graph:
    from rdflib import RDF, OWL

    g = rdflib.Graph()

    def encode_concept(s: str) -> rdflib.URIRef:
        return rdflib.URIRef(
            "#" + s.replace("%", "%25").replace(" ", "%20").replace("\n", "%20")
        )

    base = f"file:///{initial_concept.replace(" ", "_")}.owl"
    init_uri = rdflib.URIRef(base)
    g.add((init_uri, RDF.type, OWL.Ontology))

    for concept in hierarchy.keys():
        if concept == top_concept or concept == bottom_concept:
            continue
        g.add((encode_concept(concept), RDF.type, OWL.Class))
        g.add((encode_concept(concept), rdflib.RDFS.label, rdflib.Literal(concept)))
        g.add(
            (
                encode_concept(concept),
                rdflib.RDFS.comment,
                rdflib.Literal(hierarchy[concept]["definition"]),
            )
        )
        for superconcept in hierarchy[concept]["superconcepts"]:
            if superconcept == top_concept:
                continue
            g.add(
                (
                    encode_concept(concept),
                    rdflib.RDFS.subClassOf,
                    encode_concept(superconcept),
                )
            )

    return g


def output_hierarchy_as_rdf_graph(hierarchy: dict[str, dict[str, Any]], filepath):
    graph = hierarchy_to_rdf_graph(hierarchy)
    v = graph.serialize(format="pretty-xml")
    with open(f"{filepath}.owl", "w", encoding="utf8") as file:
        file.write(v)


# DOT FILE
def escape_concept(c):
    return c.replace('"', r"\"").replace("\n", " ")


def hierarchy2dot(hierarchy: dict[str, dict[str, Any]]) -> str:
    res = ""
    res += "digraph {\n"
    res += "  rankdir=LR;\n"
    for concept in hierarchy:
        if concept == bottom_concept or concept == top_concept:
            continue
        res += f' "{escape_concept(concept)}" [shape = rect'
        """
        if hierarchy[concept]["state"] == ExplorationStatus.EXPLORED_EMPTY:
            res += ", color=green"
        if hierarchy[concept]["state"] == ExplorationStatus.EXPLORED:
            res += ", color=black"
        if hierarchy[concept]["state"] == ExplorationStatus.UNEXPLORED:
            res += ", color=red"
        """
        res += "];\n"
        for subconcept in hierarchy[concept]["subconcepts"]:
            if subconcept == bottom_concept or concept == top_concept:
                continue
            res += f'  "{escape_concept(concept)}" -> "{escape_concept(subconcept)}";\n'
        for synonym in hierarchy[concept]["cycles"]:
            res += f'  "{escape_concept(concept)}" -> "{escape_concept(list(synonym.keys())[0])}"[color=blue, dir=both];\n'
        for subconcept in hierarchy[concept]["subconcepts (exceeding outdegree)"]:
            res += f'  "{escape_concept(concept)}" -> "{escape_concept(subconcept)}"[color=grey];\n'

    metadata_label = (
        "Concepts: {}\\l Subsumptions: {}\\l Exploration Depth: {}\\l Frequency Threshold: {}\\l Chosen Model: {}\\l"
    ).format(
        metadata["statistics"]["Concepts in ontology"],
        metadata["statistics"]["Subsumptions (transitive reduction)"],
        "unlimited"
        if metadata["parameters"]["exploration depth"] == 100
        else metadata["parameters"]["exploration depth"],
        metadata["parameters"]["frequency threshold for sampling"],
        chosen_model
    )

    res += "subgraph cluster_metadata {\n"
    res += 'metadata_node [label="{}" shape=none];\n'.format(metadata_label)
    res += "}\n"

    res += "}\n"
    return res


def finalize_metadata(hierarchy):
    for concept in all_discovered_concept:
        if concept.duplicate:
            metadata["duplicates"] += 1
        elif concept.verified:
            if concept.renamed_from and concept.name in hierarchy.keys():
                metadata["statistics"]["Accepted after renaming"] += 1
        elif concept.instance:
            metadata["statistics"]["Dismissed as instance"] += 1
        elif concept.partOf:
            metadata["statistics"]["Dismissed as partOf"] += 1
        elif concept.renamed_from:
            if concept.subconcept_discovery_concept is False:
                metadata["statistics"]["Dismissed as no subconcept of parent"] += 1
            if concept.subconcept_of_initial_concept is False:
                metadata["statistics"]["Dismissed as no subconcept of seed"] += 1
    metadata["statistics"]["Concepts in ontology"] = len(hierarchy) - 2
    metadata["statistics"]["Dismissed terms"] = (
        metadata["statistics"]["Dismissed as no subconcept of seed"]
        + metadata["statistics"]["Dismissed as no subconcept of parent"]
        + metadata["statistics"]["Dismissed as instance"]
        + metadata["statistics"]["Dismissed as partOf"]
    )
    metadata["statistics"]["Tokens"] = metadata["tokens (total)"]
    metadata["statistics"]["Prompts"] = metadata["prompts"]
    metadata["statistics"]["Cost of ontology in dollar"] = round(
        (
            (
                (metadata["tokens (requests)"] * model_cost_input)
                + (metadata["tokens (responses)"] * model_cost_output)
            )
            / 1000
        ),
        2,
    )  # round to 2 decimals (a cent)

    metadata["statistics"]["Cost per concept in cent"] = round(
        (
            (metadata["tokens (requests)"] * model_cost_input)
            + (metadata["tokens (responses)"] * model_cost_output)
        )
        / (10 * metadata["statistics"]["Concepts in ontology"]),
        2,
    )
    metadata["statistics"]["Prompts per concept"] = round(
        metadata["statistics"]["Prompts"]
        / metadata["statistics"]["Concepts in ontology"],
        2,
    )

    metadata["statistics"]["Tokens per concept"] = round(
        metadata["statistics"]["Tokens"]
        / metadata["statistics"]["Concepts in ontology"]
    )
    outdegrees = []
    synonyms = 0
    max_depth = 0
    max_out = 0
    max_in = 0
    in_total = 0
    out_total = 0
    no_subs = 0
    levels = []
    ignore = {top_concept, bottom_concept}
    for concept in hierarchy:
        if concept in ignore:
            continue
        if concept == initial_concept:
            hierarchy[concept]["levels"] = [0]
        else:
            hierarchy[concept]["levels"] = []
    i = 0
    flag = True
    while flag is True and i < 1000:
        flag = False
        for concept in hierarchy:
            if i in hierarchy[concept]["levels"] and concept not in ignore:
                for subconcept in hierarchy[concept]["subconcepts"]:
                    if subconcept not in ignore:
                        hierarchy[subconcept]["levels"].append(i + 1)
                        flag = True
        i += 1
    for concept in hierarchy:
        if concept in ignore:
            continue
        else:
            sup = [
                con for con in hierarchy[concept]["superconcepts"] if con not in ignore
            ]
            sub = [
                con for con in hierarchy[concept]["subconcepts"] if con not in ignore
            ]
            out_total = out_total + len(sub)
            in_total = in_total + len(sup)
            if max_out < len(sub):
                max_out = len(sub)
            if max_in < len(sup):
                max_in = len(sup)
            if len(sub) == 0:
                no_subs += 1
            if max_depth < min(hierarchy[concept]["levels"]):
                max_depth = min(hierarchy[concept]["levels"])

            min_level = min(hierarchy[concept]["levels"])
            while min_level >= len(levels):
                levels.append([])
            levels[min_level].append((len(sup), len(sub)))
            if hierarchy[concept]["cycles"] != 0:
                synonyms += len(hierarchy[concept]["cycles"])
            out = len(sub)
            while out >= len(outdegrees):
                outdegrees.append([])
            outdegrees[out].append(min_level)
    for i, level in enumerate(levels):
        if level:
            metadata["statistics"][f"Depth {i}"] = {
                "Concepts": len(level),
                "Concepts without subconcept": len([t[1] for t in level if t[1] == 0]),
                "Maximal indegree": max([t[0] for t in level])
                if len(level) != 0
                else 0,
                "Maximal outdegree": max([t[1] for t in level])
                if len(level) != 0
                else 0,
                "Average indegree": round((sum(t[0] for t in level) / len(level)), 2)
                if len(level) != 0
                else 0,
                "Average outdegree": round((sum(t[1] for t in level) / len(level)), 2)
                if len(level) != 0
                else 0,
            }
    for i, degree in enumerate(outdegrees):
        if degree:
            metadata["statistics"][f"Outdegree {i}"] = {
                "Concepts": len(degree),
                "Minimal depth": min(degree) if degree else None,
                "Maximal depth": max(degree) if degree else None,
                "Average depth": round(sum(degree) / len(degree)) if degree else None,
                "Median depth": statistics.median(degree) if degree else None,
            }

    metadata["statistics"]["Subsumptions (transitive reduction)"] = out_total
    metadata["statistics"]["Subsumptions (found by KRIS)"] = in_total - (
        metadata["statistics"]["Concepts in ontology"] - 1
    )
    metadata["statistics"]["Maximal depth"] = max_depth
    metadata["statistics"]["Synonyms"] = synonyms
    metadata["statistics"]["Maximal indegree"] = max_in
    metadata["statistics"]["Maximal outdegree"] = max_out
    metadata["statistics"]["Average degree"] = round(
        (out_total / metadata["statistics"]["Concepts in ontology"]), 2
    )
    metadata["statistics"]["Concepts without subconcept"] = no_subs

    timestamp_finish = datetime.datetime.now().replace(microsecond=0)
    metadata["duration (seconds)"] = round(
        (timestamp_finish - timestamp_start).total_seconds()
    )
    metadata["end"] = str(timestamp_finish)


def output_hierarchy_as_svg(hierarchy, filepath):
    svg_file_path = f"{filepath}"
    graph = graphviz.Source(hierarchy2dot(hierarchy))
    graph.format = "svg"
    graph.render(svg_file_path)


discovered_lock = asyncio.Lock()
all_discovered_concept = []


class ExplorationStatus(StrEnum):
    UNEXPLORED = "unexplored"
    EXPLORED = "explored"
    EXPLORED_EMPTY = "explored_empty"


class VerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    DISCARDED_BECAUSE_INSTANCE = "DISCARDED_BECAUSE_INSTANCE"
    DISCARDED_BECAUSE_NOT_A_CONCEPT = "DISCARDED_BECAUSE_NOT_A_CONCEPT"
    DISCARDED_BECAUSE_NOT_IN_DOMAIN = "DISCARDED_BECAUSE_NOT_IN_DOMAIN"
    DISCARDED_BECAUSE_NOT_SUBCONCEPT = "DISCARDED_BECAUSE_NOT_SUBCONCEPT"


@dataclass(eq=False)
class Concept:
    name: str
    definition: str = ""
    renamed_from: str = ""
    renamed_to: str = ""
    discovered_from: str = ""
    discovered_from_definition: str = ""
    discovery_depth: int = 0
    seed_concept: str = initial_concept
    seed_definition: str = ""
    verified: None | bool = False
    instance: bool | None = None
    partOf: bool | None = None
    subconcept_of_initial_concept: bool | None = None
    subconcept_discovery_concept: bool | None = None
    duplicate: bool = False
    labels: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "in_domain": {
                "labeled": False,
                "expected_result": None,
                "error_type": None,
            },
            "subconcept_of_discovery": {
                "labeled": False,
                "expected_result": None,
                "error_type": None,
            },
            "instance": {"labeled": False, "expected_result": None},
        }
    )
    prompts: list[tuple[Any, str]] = field(default_factory=list)


class PromptCache:
    Prompt = list[dict[str, str]]

    def __init__(self):
        self.path = f"query_cache/{initial_concept}_cache.txt"
        self.cache = self.load_cache()

    def load_cache(self) -> dict[str, str]:
        result = {}

        if not os.path.isfile(self.path):
            with open(self.path, "a"):
                pass

        with open(self.path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                try:
                    k, v = json.loads(line)
                    result[json.dumps(k)] = v
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line at {line_number}: {repr(line)}")
                    print(f"Error message: {e}")
                    continue

        return result

    def put_entry(self, prompt: Prompt, response: str):
        self.cache[json.dumps(prompt)] = response
        with open(f"query_cache/{initial_concept}_cache.txt", "a") as backing_file:
            json.dump((prompt, response), backing_file)
            backing_file.write("\n")

    def get_entry(self, prompt: Prompt):
        return self.cache.get(json.dumps(prompt))


pcache = PromptCache()


class DescriptionCache:
    def __init__(self):
        self.path = f"query_cache/{initial_concept}_descriptions_cache.json"
        self.cache = self.load_cache()

    def load_cache(self) -> dict[str, str]:
        if os.path.isfile(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        else:
            return {}

    def put_entry(self, concept, description=None):
        if isinstance(concept, dict):
            updated = False
            for k, v in concept.items():
                if k not in self.cache:
                    self.cache[k] = v
                    updated = True
            if updated:
                self.save_cache()
        else:
            if concept not in self.cache and description is not None:
                self.cache[concept] = description
                self.save_cache()

    def save_cache(self):
        with open(self.path, "w") as f:
            json.dump(self.cache, f, indent=4)

    def get_entry(
        self, concept: Union[str, list[str]]
    ) -> Union[str, Dict[str, str], None]:
        if isinstance(concept, list):
            return {
                k: v
                for k, v in ((k, self.cache.get(k, None)) for k in concept)
                if v is not None
            }
        else:
            return self.cache.get(concept, None)


dcache = DescriptionCache()


class SampleCache:
    Sample = list[Union[dict[str, Union[int, float]], dict[str, str]]]

    def __init__(self):
        self.path = f"query_cache/{initial_concept}_sampling_cache.txt"
        self.cache = self.load_cache()

    def load_cache(self) -> dict[str, list]:
        result = {}

        if not os.path.isfile(self.path):
            with open(self.path, "a+"):
                pass

        for line in open(self.path, "r"):
            k, v = json.loads(line)
            result[json.dumps(k)] = v
        return result

    def put_entry(self, prompt: Sample, response: list):
        self.cache[json.dumps(prompt)] = response
        backing_file = open(f"query_cache/{initial_concept}_sampling_cache.txt", "a")
        json.dump((prompt, response), backing_file)
        backing_file.write("\n")
        pass

    def get_entry(self, prompt: Sample):
        return self.cache.get(json.dumps(prompt))


scache = SampleCache()


# API CALLS TO THE openAI API
# Helper function created to perform the asynchronous HTTP POST requests to the openAI API
async def async_api_call(
    client: httpx.AsyncClient, url, headers, data, timeout: float
) -> Any:
    resp = await client.post(
        url, headers=headers, json=data, timeout=httpx.Timeout(timeout)
    )
    response = resp.json()
    return response


async def gpt_api_call(
    client,
    message_list,
    timeout_duration: float,
    max_tokens: int,
    temp=0.0,
    top_prob=0.99,
    number_choices=1,
):
    # Construct data for API request and ensure retriability
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Organization": org if org else "",
    }

    data = {
        "model": chosen_model,
        "messages": message_list,
        "temperature": temp,
        "top_p": top_prob,
        "n": number_choices,
        "max_tokens": max_tokens,
    }
    url = "https://api.openai.com/v1/chat/completions"

    try:
        resp = await async_api_call(client, url, headers, data, timeout_duration)
        return resp
    except httpx.ReadTimeout:
        log.error("ReadTimeout with timeout {}".format(timeout_duration))
    except Exception as e:
        exception_type = type(e).__name__
        log.error(
            f'API call failed with exception type "{exception_type}" and exception: {e}'
        )
    return None


# Prompt the OpenAI GPT 3.5 API
async def prompt_gpt(
    client,
    message_list: list[dict[str, str]],
    max_tokens: int,
    cache_this_get: bool = True,
    cache_this_put: bool = True,
    update_token_count: bool = True,
    temp=0.0,
    top_prob=0.99,
    number_choices=1,
    fixed_timeout=10,
) -> Union[str, None, list[str]]:
    log.info(f"Prompting {message_list}")

    if cache_this_get:
        ch = pcache.get_entry(message_list)
        if ch:
            await update_metadata_from_cache(message_list, ch)
            return ch

    number_choices_original = number_choices
    output_choices = (
        []
    )  # initialize output for number_choices > 1 if we have to requery for singular choices

    remaining_tries = 5
    delay = 1
    response = None
    while remaining_tries > 0:
        timeout_duration = max(
            (5 * (6 - remaining_tries) + 5),
            float(math.floor(max_tokens / 40)),
            fixed_timeout,
        )
        response = await gpt_api_call(
            client,
            message_list,
            timeout_duration,
            max_tokens,
            temp,
            top_prob,
            number_choices,
        )
        if response and "choices" in response:
            if update_token_count:
                await update_metadata(response)
            if number_choices_original == 1:
                if (
                    response["choices"][0]["finish_reason"] == "length"
                    and max_tokens > 1
                ):  # We don't retry with higher max_tokens for yes/no questions
                    if max_tokens == 2048:
                        log.error(f"Completion > 2048 tokens. Response: {response}")
                        output = response["choices"][0]["message"]["content"].strip()
                        log.info(
                            f"Completion exceeds model max tokens (2048). See error log for full response. Completion: {output}"
                        )
                        return None
                    else:
                        log.info(
                            f'Response bigger than expected. Retrying with higher token maximum. Response: {response["choices"][0]["message"]["content"]}'
                        )
                        max_tokens = min(max(max_tokens * 4, 100), 2048)
                else:
                    output = response["choices"][0]["message"]["content"].strip()
                    log.info(f"Response {output}")
                    if cache_this_put:
                        pcache.put_entry(message_list, output)
                    await print_and_log_prompt_response(message_list, output)
                    return output
            else:
                reprompt_counter = 0
                for choice in response["choices"]:
                    if (
                        choice["finish_reason"] == "stop" or max_tokens == 1
                    ):  # we ignore finish_reason == length for max_tokens == 1
                        output_choices.append(choice["message"]["content"])
                    else:
                        reprompt_counter += 1
                output = output_choices
                if reprompt_counter == 0:
                    log.info(f"Response: {output}")
                    await print_and_log_prompt_response(message_list, output)
                    return output
                else:
                    number_choices = reprompt_counter
                    if max_tokens == 2048:
                        log.error(f"Completion > 2048 tokens. Response: {response}")
                        log.info(
                            f"Some completion choices exceed model max tokens (2048). See error log for full response."
                        )
                        if output == []:
                            return None
                        else:
                            return output
                    else:
                        log.info(
                            f"{reprompt_counter} completion choices are bigger than expected. Retrying with higher token maximum."
                        )
                        max_tokens = min(max(max_tokens * 4, 100), 2048)
        else:
            remaining_tries -= 1
            if remaining_tries > 0:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff ()
    log.error(
        f"Failed getting a response from openAI API after 10 tries with prompt {message_list}\nFull response: {response}"
    )
    await print_and_log_prompt_response(
        message_list, "Failed getting a response from openAI API after 10 tries"
    )
    return None


# HELPERS TO FILTER AND NORMALIZE OUTPUT STRINGS OF GPT


def normalize_concept(string: str) -> str:
    string = string.strip()
    string = string.strip(".")
    if string.startswith('"') and string.endswith('"'):
        string = string[1:-1]
    string = string.strip(".")
    string.replace('"', "'")
    string = string.capitalize()
    return string


def normalize_string(string: str) -> str:
    string = string.strip()
    string = string.strip(".")
    if string.startswith('"') and string.endswith('"'):
        string = string[1:-1]
    string = string.strip(".")
    string.replace('"', "'")
    if string[0:4] == "and ":
        string = string[4:]
    string = string.capitalize()
    return string


def normalize_definition(string: str) -> str:
    string = string.strip()
    if string.startswith('"') and string.endswith('"'):
        string = string[1:-1]
    string.replace('"', "'")
    return string


# This function deals with answers where the model can't provide a proper list of subconcepts
def filter_no_subconcepts(answer_list: list[str]):
    indicator_list = [
        "None",
        "N/a",
        "does not have subcategories",
        "does not have any subcategories",
        "no subcategories",
        "do not have subcategories",
        "do not have any subcategories",
        "cannot provide",
        "cannot determine",
        "cannot complete",
        "I don't have",
        "I'm sorry",
    ]
    for answer in answer_list:
        for indicator in indicator_list:
            if indicator in answer:
                answer_list = []
    patterns = [
        re.compile(r"no\s+\w+\s+subcategories"),
        re.compile(r"no\s+\w+\s+\w+\s+subcategories"),
    ]
    for answer in answer_list:
        for pattern in patterns:
            if pattern.search(answer):
                answer_list = []
    return answer_list


def filter_mistakes(answer_list: list[str]) -> list[str]:
    for answer in answer_list:
        if (
            answer.lower()
            in [
                "more",
                "others",
                "other",
                "many more",
                "many others",
                "many other",
                "and more",
                "and others",
                "and other",
                "and many more",
                "and many others",
                "and many other",
            ]
            or "subcategory" in answer
            or "subcategories" in answer
        ):
            answer_list.remove(answer)
    return answer_list


def is_valid_list(answer):
    # Check if the string starts with a word character (alphanumeric or underscore)
    if not (
        re.match(r"^\w", answer) or answer.startswith('"') or answer.startswith("'")
    ):
        return False

    # Check if there are any line breaks in the string
    if "\n" in answer:
        return False

    # Split the string by commas
    parts = answer.split(",")

    # Check if each part starts with a space and a word character (alphanumeric or underscore)
    for part in parts[1:]:
        if not re.match(r"^ \w", part):
            return False

    # Check if each part contains no more than seven words
    for part in parts:
        if len(part.split()) > 7:
            return False
    indicator_list = [
        "cannot provide",
        "cannot determine",
        "cannot complete",
        "I don't have",
        "I'm sorry",
        "Sorry",
        "AI language model",
        "cannot perform",
        "does not make sense",
    ]
    for indicator in indicator_list:
        if indicator in answer:
            return False
    return True


def fetch_numerical_and_bulleted_lists(answer: str) -> list:
    # TODO
    return [True]


def is_valid_answer(answer):
    indicator_list = [
        "cannot provide",
        "cannot determine",
        "cannot complete",
        "cannot perform",
        "cannot name",
        "does not make sense",
        "cannot be provided",
        "cannot be named",
        "cannot be completed",
        "cannot be determined",
        "cannot be performed",
        "I don't have",
        "I'm sorry",
        "Sorry",
        "AI language model",
    ]
    for indicator in indicator_list:
        if indicator in answer:
            return False
    return True


async def concept_depth(concept: str) -> int:
    async with discovered_lock:
        if concept == initial_concept:
            return 0
        for i in all_discovered_concept:
            if i.name == concept and i.verified is not False and i.duplicate is False:
                result = int(i.discovery_depth)
                return result
        return 0


# DIFFERENT PROMPTS CHATGPT / GPT-3.5

# Constructing messages

explanation_subconcepts = '"X" is a subcategory of "Y" if every element of "X" is also an element of "Y". In other words, "X" is a subset of "Y". This means the more general term of them is "Y". "Y" is a more general category that includes "X" as a subcategory.'
explanation_subconcepts_message = {"role": "system", "content": explanation_subconcepts}


def definition_messages(concepts: list, definitions: dict) -> list[dict[str, str]]:
    if len(concepts) == 0:
        return []
    if len(concepts) > 3:
        concepts = [initial_concept, concepts[-2], concepts[-1]]
    result = []
    concepts = list(dict.fromkeys(concepts))
    for concept in concepts:
        definition_message = {
            "role": "assistant",
            "content": f"{concept}: {definitions[concept]}",
        }
        result.append(definition_message)
    return result


# Prompting for the existence of subconcepts
async def existence_subconcepts(
    client,
    concept: str,
    parents: list,
    definitions: dict,
) -> bool:
    if concept == bottom_concept:
        return False

    if len(parents) == 0:
        in_prompt = f'Are there any generally accepted subcategories of "{concept}"? Answer only with yes or no.'
    elif len(parents) == 1:
        in_prompt = f'"{concept}" is a subcategory of "{initial_concept}". Are there any generally accepted subcategories of "{concept}"? Answer only with yes or no.'
    else:
        in_prompt = f'"{parents[-1]}" is a subcategory of "{initial_concept}". "{concept}" is a subcategory of "{parents[-1]}". Are there any generally accepted subcategories of "{concept}"? Answer only with yes or no.'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages(parents + [concept], definitions)
    messages.append(prompt_message)
    output = await prompt_gpt(
        client,
        messages,
        1,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    if output is None or isinstance(output, list):
        return False

    output = normalize_string(output)
    if output not in {"Yes", "No"}:
        log.error(f'existence_subconcepts: "{output}"')
        return False
    if output == "No":
        log_verification_prompt(messages, output)

    return output == "Yes"


async def concept_definition_initial(client, initial_concept: str):
    global initial_concept_definition
    in_prompt = f'Give a very brief description of the term "{initial_concept}" without the use of examples.'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = [prompt_message]
    output = await prompt_gpt(
        client,
        messages,
        200,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    if output is None or isinstance(output, list):
        output = input(
            f'It was not possible to prompt for a definition of "{initial_concept}" from ChatGPT. Please enter a brief description of "{initial_concept}": '
        )
    output = normalize_definition(output)
    output = f"{output}"
    initial_concept_definition = output
    return output


async def subconcepts_definitions(
    client, subconcepts: list[Concept], concept: str, definitions: dict
) -> dict[str, str]:
    subconcept_names = [subconcept.name for subconcept in subconcepts]
    if prompt_caching is True:
        cached_definitions = dcache.get_entry(subconcept_names) or {}
    else:
        cached_definitions = {}
    to_fetch = [
        subconcept
        for subconcept in subconcepts
        if subconcept.name not in cached_definitions
    ]
    to_fetch_list = [
        subconcept
        for subconcept in subconcept_names
        if subconcept not in cached_definitions
    ]
    fetched_cache_list = [
        subconcept
        for subconcept in subconcept_names
        if subconcept in cached_definitions
    ]
    if prompt_caching is True and subconcept_names != []:
        if to_fetch_list:
            log.info(
                f"Getting descriptions for the following subconcepts from cache: {fetched_cache_list}. Prompting for remaining subconcepts: {to_fetch_list}"
            )
        else:
            log.info(f"Fetching all subconcepts from cache: {subconcept_names}")

    new_definitions: dict[str, str] = {}
    if to_fetch:
        in_prompt = f"Give a brief description of every term on the list, considered as a subcategory of {concept}, without the use of examples, in the following form:\nList element 1: brief description of list element 1.\nList element 2: brief description of list element 2.\n..."
        prompt_message = {"role": "user", "content": in_prompt}
        messages = definition_messages([initial_concept, concept], definitions)
        messages.append(
            {
                "role": "user",
                "content": f'List all of the most important subcategories of "{concept}" in a comma-separated format, skip explanations.',
            }
        )
        messages.append(
            {"role": "assistant", "content": ", ".join([s.name for s in to_fetch])}
        )
        messages.append(prompt_message)
        max_tokens = min(2048, max(len(to_fetch), 1) * 60)
        output = await prompt_gpt(
            client,
            messages,
            max_tokens,
            cache_this_get=False,
            cache_this_put=False,
            update_token_count=False,
        )
        if output is None or isinstance(output, list):
            substitute_definitions: dict[str, str] = {}
            for subconcept in to_fetch:
                substitute_definitions[
                    subconcept.name
                ] = f"{subconcept.name} is a subcategory of {concept}. {definitions[concept]}"
                new_definitions = substitute_definitions
        else:
            output = output.split("\n")
            for subconcept in to_fetch:
                definitions_subconcept = [
                    subconcept_def
                    for subconcept_def in output
                    if subconcept_def.lower().startswith(f"{subconcept.name.lower()}:")
                ]
                if len(definitions_subconcept) == 1:
                    definition_parts = definitions_subconcept[0].split(": ", 1)
                    new_definitions[subconcept.name] = definition_parts[1]
                    if prompt_caching is True:
                        dcache.put_entry(
                            subconcept.name, description=definition_parts[1]
                        )
                else:
                    new_definitions[
                        subconcept.name
                    ] = f"{subconcept.name} is a subcategory of {concept}. {definitions[concept]}"

    if isinstance(cached_definitions, dict):
        all_definitions = {**cached_definitions, **new_definitions}
    else:
        all_definitions = new_definitions
    await update_metadata_from_descriptions_cache(
        subconcepts, concept, definitions, all_definitions
    )
    return all_definitions


def filter_tokens(token_dict: dict) -> dict:
    token_filter_list = ["important", "Important", "1", "Some"]
    filtered_dict = {k: v for k, v in token_dict.items() if k not in token_filter_list}
    return filtered_dict


def count_occurrences(list_of_strings: list[str]):
    count_dict = {}
    for string in list_of_strings:
        if string in count_dict:
            count_dict[string] += 1
        else:
            count_dict[string] = 1
    count_dict = filter_tokens(count_dict)

    sorted_dict = {
        k: v
        for k, v in sorted(count_dict.items(), key=lambda item: (-item[1], item[0]))
    }
    return sorted_dict


# Create sample of completion tokens
async def create_completion_sample(
    client, concept: str, parents: list, definitions: dict, sample_caching=False
) -> list:
    if concept == bottom_concept:
        return []
    if concept == initial_concept:
        in_prompt = f'List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.'
    elif parents == [initial_concept]:
        in_prompt = f'"{concept}" is a subcategory of "{initial_concept}". List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.'
    else:
        in_prompt = f'"{parents[-1]}" is a subcategory of "{initial_concept}". "{concept}" is a subcategory of "{parents[-1]}". List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages([initial_concept] + parents + [concept], definitions)
    messages.append(prompt_message)

    output = []
    put_cache_entry = False
    if sample_caching is True:
        cache_entry = [
            {
                "temperature": temperature_sampling,
                "top_p": top_p_sampling,
                "n": n_completions_int,
            }
        ]
        cache_entry = cache_entry + messages
        token_list = scache.get_entry(cache_entry)
        if token_list:
            await update_metadata_from_cache_sample(messages, n_completions_int)
            output = token_list
        else:
            put_cache_entry = True
    if sample_caching is False or put_cache_entry is True:
        output = await prompt_gpt(
            client,
            messages,
            1,
            cache_this_get=False,
            cache_this_put=False,
            temp=temperature_sampling,
            top_prob=top_p_sampling,
            number_choices=n_completions_int,
        )
    if output is None:
        return []
    elif isinstance(output, str):
        if n_completions_int == 1:
            output = [output]  # Convert string to list
        else:
            return []
    elif not all(isinstance(item, str) for item in output):
        return []
    if put_cache_entry is True:
        cache_entry = [
            {
                "temperature": temperature_sampling,
                "top_p": top_p_sampling,
                "n": n_completions_int,
            }
        ]
        cache_entry = cache_entry + messages
        scache.put_entry(cache_entry, output)
    potential_tokens = count_occurrences(output)
    log.info(potential_tokens)
    starting_tokens = sorted(
        [
            token
            for token in potential_tokens
            if potential_tokens[token] >= frequency_threshold
        ]
    )
    return starting_tokens


# Prompting for a list of subconcepts
async def list_of_subconcepts(
    client, concept: str, parents: list, definitions: dict, start_token=""
) -> list[Concept]:
    if start_token == "":
        if concept == bottom_concept:
            return []
        if concept == initial_concept:
            in_prompt = f'List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.'
        elif parents == [initial_concept]:
            in_prompt = f'"{concept}" is a subcategory of "{initial_concept}". List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.'
        else:
            in_prompt = f'"{parents[-1]}" is a subcategory of "{initial_concept}". "{concept}" is a subcategory of "{parents[-1]}". List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.'
        prompt_message = {"role": "user", "content": in_prompt}
        messages = definition_messages(
            [initial_concept] + parents + [concept], definitions
        )
        messages.append(prompt_message)
        output = await prompt_gpt(
            client,
            messages,
            200,
            cache_this_get=prompt_caching,
            cache_this_put=prompt_caching,
        )
        new_subconcepts = []
        if output is None:
            return new_subconcepts
        elif isinstance(output, str):
            new_subconcepts = output.split(",")
            new_subconcepts = list(
                set(
                    [
                        normalize_string(string)
                        for string in new_subconcepts
                        if string != ""
                    ]
                )
            )
            new_subconcepts = filter_no_subconcepts(new_subconcepts)
            if not is_valid_list(output) and new_subconcepts != []:
                log.error(f'Syntax constraint violation: {prompt_message["content"]}')
                log.error(f"Syntax constraint violation: {output}")
                new_subconcepts = []
            new_subconcepts = filter_mistakes(new_subconcepts)
        elif isinstance(output, list):
            for choice in output:
                choice_subconcepts = choice.split(",")
                choice_subconcepts = list(
                    set(
                        [
                            normalize_string(string)
                            for string in choice_subconcepts
                            if string != ""
                        ]
                    )
                )
                choice_subconcepts = filter_no_subconcepts(choice_subconcepts)
                if not is_valid_list(choice) and choice_subconcepts != []:
                    choice_subconcepts = []
                choice_subconcepts = filter_mistakes(choice_subconcepts)
                for subconcept in choice_subconcepts:
                    new_subconcepts.append(subconcept)
            if new_subconcepts == []:
                log.error(f"Syntax constraint violation: {output}")
            list(set(new_subconcepts))
        new_subconcepts.sort()
        subconcept_depth = await concept_depth(concept) + 1
        return list(
            Concept(
                name=s,
                discovered_from=concept,
                discovered_from_definition=definitions[concept],
                discovery_depth=subconcept_depth,
                seed_definition=initial_concept_definition,
            )
            for s in new_subconcepts
        )
    else:
        if concept == bottom_concept:
            return []
        if concept == initial_concept:
            in_prompt = f'List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.\nStart your answer with "{start_token}".'
        elif parents == [initial_concept]:
            in_prompt = f'"{concept}" is a subcategory of "{initial_concept}". List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.\nStart your answer with "{start_token}".'
        else:
            in_prompt = f'"{parents[-1]}" is a subcategory of "{initial_concept}". "{concept}" is a subcategory of "{parents[-1]}". List all of the most important subcategories of "{concept}". Skip explanations and use a comma-separated format like this:\nImportant subcategory, Another important subcategory, Another important subcategory, etc.\nStart your answer with "{start_token}".'
        prompt_message = {"role": "user", "content": in_prompt}
        messages = definition_messages(
            [initial_concept] + parents + [concept], definitions
        )
        messages.append(prompt_message)
        output = await prompt_gpt(
            client,
            messages,
            200,
            cache_this_get=prompt_caching,
            cache_this_put=prompt_caching,
        )
        new_subconcepts = []
        if output is None:
            return new_subconcepts
        elif isinstance(output, str):
            new_subconcepts = output.split(",")
            new_subconcepts = list(
                set(
                    [
                        normalize_string(string)
                        for string in new_subconcepts
                        if string != ""
                    ]
                )
            )
            new_subconcepts = filter_no_subconcepts(new_subconcepts)
            if not is_valid_list(output) and new_subconcepts != []:
                log.error(f'Syntax constraint violation: {prompt_message["content"]}')
                log.error(f"Syntax constraint violation: {output}")
                new_subconcepts = []
            new_subconcepts = filter_mistakes(new_subconcepts)
        elif isinstance(output, list):
            for choice in output:
                choice_subconcepts = choice.split(",")
                choice_subconcepts = list(
                    set(
                        [
                            normalize_string(string)
                            for string in choice_subconcepts
                            if string != ""
                        ]
                    )
                )
                choice_subconcepts = filter_no_subconcepts(choice_subconcepts)
                if not is_valid_list(choice) and choice_subconcepts != []:
                    choice_subconcepts = []
                choice_subconcepts = filter_mistakes(choice_subconcepts)
                for subconcept in choice_subconcepts:
                    new_subconcepts.append(subconcept)
            if new_subconcepts == []:
                log.error(f"Syntax constraint violation: {output}")
            list(set(new_subconcepts))
        new_subconcepts.sort()
        subconcept_depth = await concept_depth(concept) + 1
        return list(
            Concept(
                name=s,
                discovered_from=concept,
                discovered_from_definition=definitions[concept],
                discovery_depth=subconcept_depth,
                seed_definition=initial_concept_definition,
            )
            for s in new_subconcepts
        )


def append_prompt(
    c: Concept, messages: list[dict[str, str]], output: Union[str, None, list[str]]
):
    cnt = [msg["content"] for msg in messages]
    if output and isinstance(output, str):
        c.prompts.append((cnt, output))
    else:
        c.prompts.append((cnt, "<No response>"))


# Verify that an answer is a subconcept of the initial_concept
async def verify_subconcept_initial_concept(
    client,
    subconcept: Concept,
    initial_concept: str,
    definitions: dict,
) -> bool:
    in_prompt = f'Can "{subconcept.name}" be considered a subcategory of "{initial_concept}"? Answer only with yes or no.'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages([initial_concept, subconcept.name], definitions)
    messages.append(prompt_message)
    output = await prompt_gpt(
        client,
        messages,
        1,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    append_prompt(subconcept, messages, output)

    if output is None or isinstance(output, list):
        return False
    output = normalize_string(output)

    if output not in {"Yes", "No"}:
        log.error(f'Verify subconcept of initial concept: "{output}. {in_prompt}"')
        return False

    if output == "No":
        log_verification_prompt(messages, output)

    return output == "Yes"


# This function prompts for the direction of the subconcept relationship, it is applied in the next step
async def resolve_direction(
    client, subconcept: str, concept: str, definitions: dict
) -> str | bool:
    if subconcept == concept:
        return False
    concepts_sorted = sorted([subconcept, concept])
    in_prompt = f'Consider the terms "{concepts_sorted[0]}" and "{concepts_sorted[1]}". Which of the terms is a subcategory of the other one? Answer in the following scheme: [[X]] is a subcategory of [[Y]].'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = [explanation_subconcepts_message]
    messages += definition_messages(
        [initial_concept, concepts_sorted[0], concepts_sorted[1]], definitions
    )
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        20,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    if output is None or isinstance(output, list):
        log.error(f"Failed to determine direction of subconcept relation. {in_prompt}")

        return False
    pattern = r"\[\[(.*?)\]\]"
    masked_concepts = re.findall(pattern, output)
    masked_concepts = [normalize_concept(c) for c in masked_concepts]
    if (
        len(masked_concepts) != 2
        or (
            masked_concepts[0].lower() != concept.lower()
            and masked_concepts[0].lower() != subconcept.lower()
        )
        or "]] is a subcategory of [[" not in output
    ):
        log.error(
            f'Determine direction of subconcept relation: "{output}". {in_prompt}'
        )
        return False
    else:
        log_verification_prompt(messages, output)
    if masked_concepts[0].lower() == concept.lower():
        return concept
    if masked_concepts[0].lower() == subconcept.lower():
        return subconcept
    return False


# Verification that an answer is actually a subconcept of the concept in the context of initial_concept
async def verify_subconcept_relation_with_context(
    client,
    concept: str,
    subconcept: Concept,
    initial_concept: str,
    definitions: dict,
) -> bool:
    if subconcept == concept:
        return False
    elif concept == initial_concept:
        in_prompt = f'Is "{subconcept.name}" typically understood as a subcategory of "{concept}"? Answer only with yes or no.'
        messages = definition_messages([concept, subconcept.name], definitions)
    else:
        in_prompt = f'"{concept}" is a subcategory of "{initial_concept}". Is "{subconcept.name}" typically understood as a subcategory of "{concept}"? Answer only with yes or no.'
        messages = definition_messages(
            [initial_concept, concept, subconcept.name], definitions
        )
    prompt_message = {"role": "user", "content": in_prompt}
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        1,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    append_prompt(subconcept, messages, output)

    if output is None or isinstance(output, list):
        log.error(f'Verification subconcept relation: "{output}. {in_prompt}"')
        return False
    output = normalize_string(output)
    if output not in {"Yes", "No"}:
        log.error(f'Verification subconcept relation: "{output}. {in_prompt}"')
        return False
    if output == "No":
        log_verification_prompt(messages, output)
        return False
    else:
        return True


# Verification that an answer is actually a subconcept of the concept in the context of initial_concept
async def check_subconcept_relation(
    client, concept: str, subconcept: str, initial_concept: str, definitions: dict
) -> bool:
    if concept == subconcept:  # This should never be the case with KRIS
        return True
    elif concept == initial_concept:
        in_prompt = f'Is "{subconcept}" typically understood as a subcategory of "{concept}"? Answer only with yes or no.'
        messages = definition_messages([concept, subconcept], definitions)
    else:
        in_prompt = f'"{concept}" is a subcategory of "{initial_concept}". Is "{subconcept}" typically understood as a subcategory of "{concept}"? Answer only with yes or no.'
        messages = definition_messages(
            [initial_concept, concept, subconcept], definitions
        )
    prompt_message = {"role": "user", "content": in_prompt}
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        1,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    if output is None or isinstance(output, list):
        log.error(f'Insertion subconcept relation: "{output}. {in_prompt}"')
        return False
    output = normalize_string(output)
    if output not in {"Yes", "No"}:
        log.error(f'Insertion subconcept relation: "{output}. {in_prompt}"')
        return False
    if output == "No":
        return False
    else:
        return True


# This function tries to rename a concept once if one of the verification steps fails
async def rename_subconcept(
    client, concept: str, subconcept: Concept, initial_concept: str, definitions: dict
) -> str:
    if concept != initial_concept:
        in_prompt = f'"{concept}" is a subcategory of "{initial_concept}". The following description outlines the characteristics of a subcategory of "{concept}". Provide a concise and unambiguous name for it. Provide only the name without any explanation. Description: {definitions[subconcept.name]}'
    else:
        in_prompt = f'The following description outlines the characteristics of a subcategory of "{concept}". Provide a concise and unambiguous name for it. Provide only the name without any explanation. Description: {definitions[subconcept.name]}'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages([initial_concept, concept], definitions)
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        25,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    append_prompt(subconcept, messages, output)
    if output is None or isinstance(output, list):
        return subconcept.name
    output = normalize_concept(output)
    if is_valid_answer(output) is False:
        log.error(f'Syntax constraint violation: {prompt_message["content"]}')
        log.error(f"Syntax constraint violation: {output}")
        verification_log.info(f"Renaming failed: {subconcept} to {output}")
        return subconcept.name
    else:
        verification_log.info(f"Info:\n{subconcept.name} renamed to {output}")
    return output


def log_verification_prompt(messages, output):
    log_entry = "Prompt:\n"
    for message in messages:
        log_entry = log_entry + f'{message["content"]}\n'
    log_entry = log_entry + f"{output}"
    verification_log.info(log_entry)


async def is_instance(
    client, initial_concept: str, subconcept: Concept, definitions: dict
) -> bool:
    in_prompt = f'Is "{subconcept.name}" a specific instance or a subcategory of the category "{initial_concept}"? Answer only with "Instance" or "Subcategory".'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages([initial_concept, subconcept.name], definitions)
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        4,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    append_prompt(subconcept, messages, output)
    if output is None or isinstance(output, list):
        return False
    output = normalize_string(output)
    if output not in {"Subcategory", "Instance"}:
        log.error(f'Instance check: "{output}. {in_prompt}"')
        return False
    if output == "Instance":
        log_verification_prompt(messages, output)
    return output == "Instance"


async def is_part_of(
    client, initial_concept: str, subconcept: Concept, definitions: dict
) -> bool:
    in_prompt = f'Is "{subconcept.name}" a part or a subcategory of the category "{initial_concept}"? Answer only with "Part" or "Subcategory".'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages([initial_concept, subconcept.name], definitions)
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        4,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    append_prompt(subconcept, messages, output)
    if output is None or isinstance(output, list):
        return False
    output = normalize_string(output)
    if output not in {"Subcategory", "Part"}:
        log.error(f'PartOf check: "{output}. {in_prompt}"')
        return False
    if output == "Part":
        log_verification_prompt(messages, output)
    return output == "Part"


async def determine_cycle(
    client, concept1: str, concept2: str, initial_concept: str, definitions: dict
) -> bool:
    concepts_sorted = sorted([concept1, concept2])
    in_prompt = f'In the context of {initial_concept}, are "{concepts_sorted[0]}" and "{concepts_sorted[1]}" typically used interchangeably? Answer in one word, yes or no.'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages(
        [initial_concept, concepts_sorted[0], concepts_sorted[1]], definitions
    )
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        1,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    if output is None or isinstance(output, list):
        return False
    output = normalize_string(output)
    if output not in {"Yes", "No"}:
        log.error(f'Cycle check synonymity: "{output}. {in_prompt}"')
        return False
    return output == "Yes"


async def determine_plural(
    client, concept1: str, concept2: str, initial_concept: str, definitions: dict
) -> bool:
    concepts_sorted = sorted([concept1, concept2])
    in_prompt = f'Consider the terms "{concepts_sorted[0]}" and "{concepts_sorted[1]}". Is one of the terms the plural form of the other one? Answer in one word, yes or no.'
    prompt_message = {"role": "user", "content": in_prompt}
    messages = definition_messages(
        [initial_concept, concepts_sorted[0], concepts_sorted[1]], definitions
    )
    messages.append(prompt_message)

    output = await prompt_gpt(
        client,
        messages,
        1,
        cache_this_get=prompt_caching,
        cache_this_put=prompt_caching,
    )
    if output is None or isinstance(output, list):
        return False
    output = normalize_string(output)
    if output not in {"Yes", "No"}:
        log.error(f'Plural check synonymity: "{output}. {in_prompt}"')
        return False
    return output == "Yes"


def get_immediate_parent(concept) -> str | None:
    for i in all_discovered_concept:
        if i.name == concept and i.verified is not False and i.duplicate is False:
            return i.discovered_from


async def compute_parent_list(
    hierarchy: dict[str, dict[str, list[Any]]], concept: str
) -> list[str]:
    immediate_parent = get_immediate_parent(concept)
    if immediate_parent == None or immediate_parent == "":
        return []

    parents = [immediate_parent]
    while initial_concept not in parents:
        for i in range(len(hierarchy[parents[-1]]["superconcepts"])):
            if hierarchy[parents[-1]]["superconcepts"][i] not in parents:
                parents.append(hierarchy[parents[-1]]["superconcepts"][i])
                break
            if i == len(hierarchy[parents[-1]]["superconcepts"]):
                parents.append(initial_concept)
    parents.reverse()
    return parents


def definitions_list(
    hierarchy: dict[str, dict[str, list[Any]]], concepts: list[str]
) -> dict:
    definitions = {}
    for concept in concepts:
        definitions[concept] = hierarchy[concept]["definition"]
    return definitions


async def verification_checks(
    client, concept: str, initial_concept: str, definitions, subconcept: Concept
):
    subconcept_of_initial_concept = await verify_subconcept_initial_concept(
        client, subconcept, initial_concept, definitions
    )
    subconcept.subconcept_of_initial_concept = subconcept_of_initial_concept
    if not subconcept_of_initial_concept:
        subconcept.verified = False
        return

    subconcept_discovery_concept = await verify_subconcept_relation_with_context(
        client, concept, subconcept, initial_concept, definitions
    )
    subconcept.subconcept_discovery_concept = subconcept_discovery_concept
    if not subconcept_discovery_concept:
        subconcept.verified = False
        return

    subconcept.verified = True
    return


async def verify_subconcept(
    client, concept: str, definitions, subconcept: Concept, subconcepts
) -> None:
    if await is_instance(client, initial_concept, subconcept, definitions):
        subconcept.instance = True
        subconcept.verified = False
        subconcept.partOf = False
        return
    subconcept.instance = False

    if await is_part_of(client, initial_concept, subconcept, definitions):
        subconcept.partOf = True
        subconcept.verified = False
        return
    subconcept.partOf = False

    await verification_checks(client, concept, initial_concept, definitions, subconcept)

    if subconcept.verified:
        return

    renamed_subconcept = await rename_subconcept(
        client, concept, subconcept, initial_concept, definitions
    )
    if renamed_subconcept == subconcept.name:
        subconcept.renamed_to = renamed_subconcept
        subconcept.renamed_from = renamed_subconcept
        return

    subconcept.renamed_to = renamed_subconcept

    new_concept = Concept(
        name=renamed_subconcept,
        discovered_from=subconcept.discovered_from,
        discovered_from_definition=subconcept.discovered_from_definition,
        renamed_from=subconcept.name,
        definition=subconcept.definition,
        discovery_depth=subconcept.discovery_depth,
        seed_definition=subconcept.seed_definition,
    )
    subconcepts.append(new_concept)

    definitions[new_concept.name] = definitions[subconcept.name]

    await verification_checks(
        client, concept, initial_concept, definitions, new_concept
    )

    return


async def verify_subconcepts(
    client, concept: str, definitions, subconcepts: list[Concept]
) -> None:
    tasks = []
    for subconcept in subconcepts:

        async def verify_task(sub: Concept):
            await verify_subconcept(client, concept, definitions, sub, subconcepts)

        tasks.append(verify_task(subconcept))

    await asyncio.gather(*tasks)


is_subconcept_cache: dict[tuple[str, str], bool] = {}


def add_assumed_rels_to_cache(hierarchy, concept: str, subconcept: str):
    for super in hierarchy.keys():
        if concept in reachable(hierarchy, super):
            is_subconcept_cache[(super, subconcept)] = True
    return


async def is_subconcept(
    client,
    hierarchy: dict,
    definitions: dict,
    super: str,
    sub: str,
) -> bool:
    if super == bottom_concept or sub == top_concept or sub == initial_concept:
        return False
    if sub == bottom_concept or super == top_concept or super == initial_concept:
        return True

    if (super, sub) in is_subconcept_cache:
        return is_subconcept_cache[(super, sub)]

    missing_definitions = [
        i for i in [initial_concept, super, sub] if i not in definitions.keys()
    ]
    definitions.update(definitions_list(hierarchy, missing_definitions))

    res = await check_subconcept_relation(
        client,
        super,
        sub,
        initial_concept,
        definitions,
    )

    is_subconcept_cache[(super, sub)] = res

    return res


def succs(hierarchy, x):
    for y in hierarchy[x]["subconcepts"]:
        yield y


def preds(hierarchy, x):
    for y in hierarchy[x]["superconcepts"]:
        yield y


async def enhanced_top_subs(client, hierarchy, definitions, super, sub) -> bool:
    for b1 in preds(hierarchy, super):
        if not await enhanced_top_subs(client, hierarchy, definitions, b1, sub):
            return False
    return await is_subconcept(client, hierarchy, definitions, super, sub)


async def enhanced_top_sub_tasks(
    client, hierarchy, definitions, super, sub
) -> str | None:
    if await enhanced_top_subs(client, hierarchy, definitions, super, sub):
        return super
    return None


async def top_search(client, hierarchy, definitions, c: str, x: str) -> set[str]:
    pos_succ = set()

    tasks: list[asyncio.Task[str | None]] = []
    L = []
    for y in succs(hierarchy, x):
        tasks.append(
            asyncio.create_task(
                enhanced_top_sub_tasks(client, hierarchy, definitions, y, c)
            )
        )
    L = await asyncio.gather(*tasks)

    for s in L:
        if s:
            pos_succ.add(s)

    if len(pos_succ) == 0:
        return {x}

    result: set[str] = set()
    for y in pos_succ:
        result = result | await top_search(client, hierarchy, definitions, c, y)
    return result


def reachable(hierarchy, c: str) -> set[str]:
    res = set([c])
    change = True
    while change:
        change = False
        nres = set()
        for x in res:
            for y in succs(hierarchy, x):
                if y not in res:
                    nres.add(y)
                    change = True
        res |= nres
    return res


async def enhanced_bot_subs(client, hierarchy, definitions, super, sub) -> bool:
    for b2 in succs(hierarchy, sub):
        if not await enhanced_bot_subs(client, hierarchy, definitions, super, b2):
            return False
    return await is_subconcept(client, hierarchy, definitions, super, sub)


async def enhanced_bot_sub_tasks(
    client, hierarchy, definitions, super, sub
) -> str | None:
    if await enhanced_bot_subs(client, hierarchy, definitions, super, sub):
        return sub
    return None


async def bot_search(client, hierarchy, definitions, c, x, superconcepts) -> set[str]:
    candidates = set.intersection(
        *[reachable(hierarchy, superconcept) for superconcept in superconcepts]
    )

    tasks = []
    for y in preds(hierarchy, x):
        if y in candidates:
            tasks.append(
                asyncio.create_task(
                    enhanced_bot_sub_tasks(client, hierarchy, definitions, c, y)
                )
            )

    L = await asyncio.gather(*tasks)
    pos_pred = set()
    for s in L:
        if s:
            pos_pred.add(s)

    if len(pos_pred) == 0:
        return {x}

    result = set()
    for y in pos_pred:
        result = result | await bot_search(
            client, hierarchy, definitions, c, y, superconcepts
        )

    return result


async def add_to_hierarchy(
    client,
    hierarchy: dict[str, dict[str, Any]],
    concept: str,
    definitions: dict,
) -> dict[str, dict[str, Any]]:
    superconcepts = await top_search(
        client, hierarchy, definitions, concept, top_concept
    )
    subconcepts = await bot_search(
        client, hierarchy, definitions, concept, bottom_concept, superconcepts
    )

    # Check if there is a cycle
    subconcepts_copy = subconcepts.copy()
    superconcepts_copy = superconcepts.copy()
    for subconcept in subconcepts_copy:
        r = reachable(hierarchy, subconcept)
        for superconcept in superconcepts_copy:
            if superconcept in r:
                if (
                    superconcept == subconcept
                ):  # this is only the case if this is a 2-cycles
                    if superconcept not in definitions.keys():
                        definitions.update(definitions_list(hierarchy, [superconcept]))
                    if superconcept == concept:
                        is_cycle = True
                    else:
                        is_cycle = await determine_cycle(
                            client, concept, superconcept, initial_concept, definitions
                        )
                        if not is_cycle:
                            is_cycle = await determine_plural(
                                client,
                                concept,
                                superconcept,
                                initial_concept,
                                definitions,
                            )
                    if is_cycle:
                        hierarchy[superconcept]["cycles"].append(
                            {concept: definitions[concept]}
                        )
                        return hierarchy  # instead of adding concept to the hierarchy we add it as a cycle to superconcept
                    else:
                        sub = await resolve_direction(
                            client, concept, superconcept, definitions
                        )
                        if sub is False:
                            sub = concept
                        if sub == superconcept:
                            superconcepts.remove(superconcept)
                            for s in hierarchy[superconcept]["superconcepts"]:
                                superconcepts.add(s)
                        elif sub == concept:
                            subconcepts.remove(superconcept)
                            for s in hierarchy[superconcept]["subconcepts"]:
                                subconcepts.add(s)
                        else:
                            return hierarchy
                else:  # longer cycle, absolute edge case that never happened so far, if this happens we don't accept the concept in our ontology
                    return hierarchy

    hierarchy[concept] = {
        "superconcepts": [],
        "levels": [],
        "subconcepts": [],
        "cycles": [],
        "subconcepts (exceeding outdegree)": [],
        "definition": definitions[concept],
        "state": ExplorationStatus.UNEXPLORED,
    }
    # Add arrows from superconcepts to concept
    for superconcept in superconcepts:
        level = min(hierarchy[superconcept]["levels"])
        hierarchy[concept]["levels"].append(level + 1)
        hierarchy[superconcept]["subconcepts"].append(concept)
        hierarchy[concept]["superconcepts"].append(superconcept)

        # Remove arrows from superconcepts to subconcepts
        for subconcept in subconcepts:
            if subconcept in hierarchy[superconcept]["subconcepts"]:
                hierarchy[superconcept]["subconcepts"].remove(subconcept)
                hierarchy[subconcept]["superconcepts"].remove(superconcept)
                if subconcept != bottom_concept:
                    try:
                        ind = hierarchy[subconcept]["levels"].index(level + 1)
                        hierarchy[subconcept]["levels"][ind] = level + 2
                    except ValueError:
                        pass

    # Add arrows from concept to subconcepts
    for subconcept in subconcepts:
        hierarchy[subconcept]["superconcepts"].append(concept)
        hierarchy[concept]["subconcepts"].append(subconcept)

    return hierarchy


# This function creates an image output for every concept which is added to the hierarchy
def dump_hierarchy(hierarchy: dict[str, dict[str, Any]]):
    svg_file_path = os.path.join(
        ontology_directory, "svg_dump", f"{filename}-{len(hierarchy)}"
    )
    graph = graphviz.Source(hierarchy2dot(hierarchy))
    graph.format = "svg"
    graph.render(svg_file_path, cleanup=True)


# This is the async lock for the hierarchy
h_lock = asyncio.Lock()


def should_kris_continue(hierarchy) -> bool:
    current_cost = (metadata["tokens (responses)"] * model_cost_output / 1000) + (
        metadata["tokens (requests)"] * model_cost_input / 1000
    )
    current_concept_number = len(hierarchy) - 2
    running_time = round(
        round(
            (
                datetime.datetime.now().replace(microsecond=0) - timestamp_start
            ).total_seconds()
        )
        / 60
    )

    if current_cost >= cost_threshold:
        log.info(f"Stopping due to cost {current_cost}")
        return False

    if current_concept_number >= concept_threshold:
        log.info(f"Stopping due to {current_concept_number} concepts")
        return False

    if running_time >= running_time_threshold:
        log.info(f"Stopping due to running time exceeded")
        return False

    return True


async def prompt_for_subconcepts(
    client, concept, parents, definitions
) -> list[Concept]:
    res: list[Concept] = []
    starting_tokens = await create_completion_sample(
        client, concept, parents, definitions, sample_caching=True
    )
    for token in starting_tokens:
        p = await list_of_subconcepts(
            client, concept, parents, definitions, start_token=token
        )
        if p:
            for c in p:
                if c.name not in {d.name for d in res}:
                    res.append(c)
    return res


async def explore_concept(
    client, hierarchy: dict[str, dict[str, Any]], concept: str
) -> tuple[list[Concept], dict[str, str]]:
    parents = await compute_parent_list(hierarchy, concept)
    definitions = definitions_list(
        hierarchy, list(dict.fromkeys([initial_concept] + parents + [concept]))
    )
    has_subconcepts = await existence_subconcepts(client, concept, parents, definitions)
    if not has_subconcepts:
        return [], definitions

    subconcepts = await prompt_for_subconcepts(client, concept, parents, definitions)

    # For long lists of subconcepts, we make sure to not exceed the maximal token limit for answers (2048) of the model in use
    subconcepts_partitions = [
        subconcepts[i : i + 30] for i in range(0, len(subconcepts), 30)
    ]
    for partition in subconcepts_partitions:
        partition_definitions = await subconcepts_definitions(
            client, partition, concept, definitions
        )
        definitions.update(partition_definitions)

    await verify_subconcepts(client, concept, definitions, subconcepts)

    result = []
    async with discovered_lock:
        all_verified_names = {
            i.name for i in all_discovered_concept if i.verified and not i.duplicate
        }
        for subconcept in subconcepts:
            if subconcept.name in all_verified_names:
                subconcept.duplicate = True
                log.error(f"Already discovered subconcept {subconcept.name}.")
            if subconcept.verified and subconcept.duplicate is False:
                result.append(subconcept)
            all_discovered_concept.append(subconcept)
    return result, definitions


Hierarchy = dict[str, dict[str, Any]]


async def kris_alg_task(
    client,
    exploration_queue: list[str],
    visited_concepts: set[str],
    hierarchy: Hierarchy,
    concept: str,
):
    if not should_kris_continue(hierarchy):
        return

    print_with_timestamp(f"starting to explore the concept {concept}")

    current_level = min(hierarchy[concept]["levels"])

    verified_subconcepts, definitions = await explore_concept(
        client, hierarchy, concept
    )
    if max_outdegree < len(verified_subconcepts):
        random_integers = random.sample(range(len(verified_subconcepts)), max_outdegree)
        further_exploration = []
        no_exploration = []
        further_exploration = [verified_subconcepts[i] for i in random_integers]
        no_exploration = [
            verified_subconcepts[i].name
            for i in range(len(verified_subconcepts))
            if i not in random_integers
        ]
        verified_subconcepts = further_exploration
        hierarchy[concept]["subconcepts (exceeding outdegree)"] = no_exploration

    print_with_timestamp(
        f"obtained {len(verified_subconcepts)} verified subconcepts of {concept}"
    )
    for subconcept in verified_subconcepts:
        if not should_kris_continue(hierarchy):
            break
        if subconcept.name in hierarchy:
            subconcept.duplicate = True
            log.error(
                f"Discovered subconcept {subconcept.name} twice, skipping for now"
            )
            continue

        add_assumed_rels_to_cache(hierarchy, concept, subconcept.name)

        # We don't use the result of this, we just use it to fill the cache in parallel
        superconcepts = await top_search(
            client, hierarchy, definitions, subconcept.name, initial_concept
        )
        await bot_search(
            client,
            hierarchy,
            definitions,
            subconcept.name,
            bottom_concept,
            superconcepts,
        )

        async with h_lock:
            # Add information to hierarchy
            hierarchy = await add_to_hierarchy(
                client, hierarchy, subconcept.name, definitions
            )
            # dump_hierarchy(hierarchy)

        if (
            subconcept.name in hierarchy
            and current_level + 1 < exploration_depth
            and subconcept.name not in visited_concepts
        ):
            visited_concepts.add(subconcept.name)
            exploration_queue.append(subconcept.name)

    if len(verified_subconcepts) == 0:
        hierarchy[concept]["state"] = ExplorationStatus.EXPLORED_EMPTY
    else:
        hierarchy[concept]["state"] = ExplorationStatus.EXPLORED

    print_with_timestamp(f"finished exploring the concept {concept}")


async def kris_alg(client, hierarchy: dict[str, dict[str, Any]]):
    hierarchy.update(
        {
            top_concept: {
                "superconcepts": [],
                "levels": [-1],
                "subconcepts": [initial_concept],
                "cycles": [],
                "subconcepts (exceeding outdegree)": [],
                "state": ExplorationStatus.EXPLORED,
                "definition": "top concept",
            },
            initial_concept: {
                "superconcepts": [top_concept],
                "levels": [0],
                "subconcepts": [bottom_concept],
                "cycles": [],
                "subconcepts (exceeding outdegree)": [],
                "definition": "",
                "state": ExplorationStatus.UNEXPLORED,
            },
            bottom_concept: {
                "superconcepts": [initial_concept],
                "levels": [99],
                "subconcepts": [],
                "cycles": [],
                "subconcepts (exceeding outdegree)": [],
                "state": ExplorationStatus.EXPLORED,
                "definition": "bottom concept",
            },
        }
    )
    hierarchy[initial_concept]["definition"] = await concept_definition_initial(
        client, initial_concept
    )
    seed_concept = Concept(
        name=initial_concept,
        definition=hierarchy[initial_concept]["definition"],
        seed_definition=hierarchy[initial_concept]["definition"],
        verified=None,
    )
    all_discovered_concept.append(seed_concept)

    visited_concepts = set()
    exploration_queue = [initial_concept]
    while should_kris_continue(hierarchy) and len(exploration_queue) > 0:
        concept = exploration_queue.pop(0)
        await kris_alg_task(
            client, exploration_queue, visited_concepts, hierarchy, concept
        )


# Run the main coroutine and save the result
def main():
    timeout = httpx.Timeout(10.0)
    hierarchy = {}

    async def main_async():
        async with httpx.AsyncClient(timeout=timeout) as client:
            await kris_alg(client, hierarchy)
        output_hierarchy_and_metadata(hierarchy, filepath)

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt as a:
        print("run aborted")
        output_hierarchy_and_metadata(hierarchy, filepath)


def output_hierarchy_and_metadata(hierarchy, filepath):
    finalize_metadata(hierarchy)

    # THE CREATION OF FINAL OUTPUT FILES
    output_hierarchy_as_json(hierarchy, filepath)
    output_hierarchy_as_svg(hierarchy, filepath)
    output_hierarchy_as_rdf_graph(hierarchy, filepath)

    # Save metadata.json
    with open(f"{ontology_directory}/{filename}_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)
    """
    # Save all discovered concepts
    with open(f"{filepath}-concepts.json", "w", encoding="utf8") as file:
        json.dump(all_discovered_concept, file, indent=4, cls=DataclassJSONEncoder)
    """


if __name__ == "__main__":
    main()
