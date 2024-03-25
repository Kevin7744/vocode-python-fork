import argparse
import json
import os
import copy
import random
from datetime import datetime

from vocode.utils.campaign_utils.utils import get_campaign_configs
from vocode.utils.campaign_utils.campaign import BusinessCampaign

AWS_PROFILE_NAME = os.getenv("AWS_PROFILE_NAME", "marrlabsmfa")
# Get the current time to timestamp the log file
current_time = datetime.now()
# Format the timestamp. Example: '2024-01-23_15-30-00'
timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

def parse_opts():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="""Generate 
                                     an experiment config and the data needed to run main.py.""")
    
    parser.add_argument("--exp-config-template", 
                        help="the template for the experiment config", 
                        required=True)
    parser.add_argument("--campaign-name", 
                        help="the name of the campaign we want to simulate", 
                        required=True)
    parser.add_argument("--model-name", 
                        help="the name of the gpt model to use", 
                        choices=["gpt-3.5-turbo", "gpt-4-1106-preview"],
                        default="gpt-3.5-turbo")
    parser.add_argument("--business-name", 
                        help="the business name needed for campaign",
                        required=True)
    parser.add_argument("--human-params", 
                        help="json file with human description", 
                        required=True)
    parser.add_argument("--human-config-template", 
                        help="json file with human config template", 
                        required=True)
    parser.add_argument("--dyn-params", 
                        help="json file with dyn params", 
                        required=True)
    parser.add_argument("--out-dir", 
                        help="output dir", 
                        required=True)

    # Parse arguments
    args = parser.parse_args()
    return args

def check_file(filename: str) -> bool:
    """
    Check that a file exists and is 
    not empty
    """
    if not os.path.isfile(filename):
        return False

    # Check if the file is not empty
    return os.path.getsize(filename) > 0

def load_json_file(filename):
    """
    Load and return the data from a JSON file.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data
    except json.JSONDecodeError:
        raise RuntimeError(f"The file {filename} is not a valid JSON file.")

def write_to_file(json_data, base_dir, dyn_params, 
                  campaign_name, business_name):
        # set up the output dir
    dir_prefix = '_'.join(str(value) for key, 
                          value in sorted(dyn_params.items()) if key not in ["AgentAvailability"])

    output_dir = os.path.join(base_dir, 
                              campaign_name, 
                              business_name, 
                              dir_prefix
                              )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config_name = os.path.join(output_dir, 
                               f"config.{timestamp}.json")
    print(config_name)
    try:
        with open(config_name, 'w') as file:
            json.dump(json_data, file, indent=4)
        print(f"Data successfully written to {config_name}")
    except IOError as e:
        raise RuntimeError(f"Error writing to file: {e}")

def main():
    args = parse_opts()
    
    exp_config_template_filename = args.exp_config_template
    human_params_filename = args.human_params
    human_config_template_filename = args.human_config_template
    dyn_param_filename = args.dyn_params

    model_name = args.model_name
    campaign_name = args.campaign_name
    business_name = args.business_name

    base_dir = args.out_dir 
        
    for f in [exp_config_template_filename, 
              human_params_filename,
              human_config_template_filename,
              dyn_param_filename]:
        if not check_file(f):
            raise RuntimeError(f"{f} does not exist")
        
    exp_config_template = load_json_file(exp_config_template_filename)
    human_params = load_json_file(human_params_filename)
    human_config_template = load_json_file(human_config_template_filename)
    dyn_params = load_json_file(dyn_param_filename)

    assert campaign_name in human_params
    assert campaign_name in dyn_params

    human_params_value = random.choice(human_params[campaign_name])
    dyn_params_value = random.choice(dyn_params[campaign_name])
    
    # Get the Agent config
    campaign_configs = get_campaign_configs("outboundCampaign-campaignConfigs", 
                                            AWS_PROFILE_NAME)
    agent_config = campaign_configs.get(campaign_name)
    agent_config_obj = BusinessCampaign.from_dict(agent_config)
    agent_config_obj.dyn_params = dyn_params_value

    # Get the human config
    human_llm_prompt = human_config_template["LLMPrompt"]
    for k, v in human_params_value.items():
        human_llm_prompt = human_llm_prompt.replace(f"<{k}>", v)
    for k, v in dyn_params_value.items():
        human_llm_prompt = human_llm_prompt.replace(f"<{k}>", v)
    human_llm_prompt = human_llm_prompt.replace("<BusinessName>", 
                                                business_name)

    config_template = copy.copy(exp_config_template)
    config_template["business_name"] = business_name
    config_template["campaign_name"] = campaign_name
    config_template["agent_gpt_config"]["gpt_model_name"] = model_name
    config_template["agent_gpt_config"]["initial_message"] = (agent_config_obj
                                                              .first_prompt_for_biz(business_name))
    config_template["agent_gpt_config"]["prompt_preamble"] = (agent_config_obj
                                                              .llm_prompt_for_biz(business_name))

    config_template["human_gpt_config"]["gpt_model_name"] = model_name
    config_template["human_gpt_config"]["prompt_preamble"] = human_llm_prompt

    write_to_file(config_template, base_dir, 
                  dyn_params_value, campaign_name, 
                  business_name)
    
if __name__ == "__main__":
    main()
    