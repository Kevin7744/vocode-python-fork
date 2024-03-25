import argparse
import asyncio
from datetime import datetime
import json
import os

from vocode.streaming.agent import *
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.utils import create_conversation_id
from vocode.utils.setenv_utils import set_env
from vocode.streaming.models.transcript import Transcript

AWS_PROFILE_NAME = os.getenv("AWS_PROFILE_NAME", "marrlabsmfa")

def parse_opts():
    parser = argparse.ArgumentParser(description="Run an agent simulation with a config file")
    parser.add_argument('config', type=str, help='Path to configuration file')
    args = parser.parse_args()
    return args

async def async_main(config_name):
    set_env(aws_profile_name=AWS_PROFILE_NAME)

    # Load the configuration file
    with open(config_name, 'r') as config_file:
        config = json.load(config_file)

    agent_llm = get_agent_llm(config)
    
    client_llm = get_client_llm(config)

    conversation_id = create_conversation_id()
    initial_agent_response = agent_llm.get_agent_config().initial_message
    print(f"\nAgent: {initial_agent_response.text}")
    agent_response_str = initial_agent_response.text
    is_interrupt = False
    agent_llm.transcript.add_bot_message(initial_agent_response.text,
                                        conversation_id=conversation_id) 
    client_llm.transcript.add_human_message(initial_agent_response.text,
                                        conversation_id=conversation_id) 

    while True:   
        # Get human response
        human_response = client_llm.generate_response(agent_response_str, 
                                                    conversation_id,
                                                    is_interrupt=is_interrupt) 
        human_response_str = ""
        async for response, _ in human_response:
            human_response_str += response
        print(f"\nHuman: {human_response_str}")
        agent_llm.transcript.add_human_message(human_response_str,
                                        conversation_id=conversation_id) 
        client_llm.transcript.add_bot_message(human_response_str,
                                        conversation_id=conversation_id) 

        if client_llm.agent_config.end_conversation_on_goodbye:
            user_goodbye_detected_task = client_llm.create_goodbye_detection_task(
                                human_response_str)
            if await user_goodbye_detected_task:
                break

        # Get agent response
        agent_response = agent_llm.generate_response(human_response_str, 
                                                    conversation_id,
                                                    is_interrupt=is_interrupt) 
        agent_response_str = ""
        async for response, _ in agent_response:
            agent_response_str += response
        print(f"\nAgent: {agent_response_str}")
        agent_llm.transcript.add_bot_message(agent_response_str,
                                        conversation_id=conversation_id) 
        client_llm.transcript.add_human_message(agent_response_str,
                                        conversation_id=conversation_id) 

        if agent_llm.agent_config.end_conversation_on_goodbye:
            agent_goodbye_detected_task = agent_llm.create_goodbye_detection_task(
                                agent_response_str)
            if await agent_goodbye_detected_task:
                break


    write_to_file(config_name, agent_llm, config)
    
    agent_llm.terminate()
    client_llm.terminate()

    print(agent_llm.transcript.to_string())

def write_to_file(config_name, agent_llm, config):
    """
    Write to output json file in the same dir as 
    the input config file
    """
    # Get the current time to timestamp the log file
    current_time = datetime.now()
    # Format the timestamp. Example: '2024-01-23_15-30-00'
    timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    out_dir = os.path.dirname(config_name)
    filename = os.path.join(out_dir, f"transcript.{timestamp}.json")
    json_data = {"config": config,
                 "transcript": agent_llm.transcript.to_string()}
    try:
        with open(filename, 'w') as file:
            json.dump(json_data, file, indent=4)
        print(f"Data successfully written to {filename}")
    except IOError as e:
        raise RuntimeError(f"Error writing to file: {e}")

    
def get_client_llm(config):
    client_llm_config = ChatGPTAgentConfig(
                    generate_responses=True,
                    model_name=config["human_gpt_config"]["gpt_model_name"],
                    max_tokens=1000, 
                    prompt_preamble=config["human_gpt_config"]["prompt_preamble"],
                    end_conversation_on_goodbye=True
                )
    client_llm = ChatGPTAgent(client_llm_config)
    client_llm.is_first_response = False 
    user_transcript = Transcript()
    client_llm.attach_transcript(user_transcript)
    return client_llm

def get_agent_llm(config):
    agent_llm_config =  ChatGPTAgentConfig(
     initial_message=BaseMessage(text=config["agent_gpt_config"]["initial_message"]),
     prompt_preamble=config["agent_gpt_config"]["prompt_preamble"],
     generate_responses=True,
     end_conversation_on_goodbye=True,
     max_tokens=1000, 
     model_name=config["agent_gpt_config"]["gpt_model_name"],
   )
    agent_llm = ChatGPTAgent(agent_llm_config)
    agent_llm.is_first_response = False 
    agent_transcript = Transcript()
    agent_llm.attach_transcript(agent_transcript)
    return agent_llm
    
if __name__ == "__main__":
    args = parse_opts()
    asyncio.run(async_main(args.config))