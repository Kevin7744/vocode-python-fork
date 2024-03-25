# Agent Simulator

Script to run a GPT simulation between two GPTs. One representing the agent and 
one representing the human. This is meant to help us iterate faster on our GPT prompts. 

## Code needed
* `create_simulation_config.py`: this is used to create the experiment config to run the simulation.

* `main.py`: this will run the actual simulation.

## Files needed
* human_params.json: this json is a dictionary keyed by the campaign_name. The campaign_name needs to exist in the dynamodb table `outboundCampaign-campaignConfigs`. For each campaign that you wish to simulate, include the needed human parameters in this json, for example: their name, their style in conversing, etc.
* dyn_params.json: this json is a dictionary keyed by the campaign_name. The campaign_name needs to exist in the dynamodb table `outboundCampaign-campaignConfigs`. For each simulation that you wish to simulate, include the needed values for any dynamic parameter.
* template_human_config_\*.json: this is a template json for the human and it contains mostly the llm prompt with some place holders that will be replaced by the contents in human_params.json.

## Simple run example with one simulation
### Generate the config for the experiment simulation
The `--campaign-name` argument needs to be a campaign that exists in the dynamodb table `outboundCampaign-campaignConfigs`. 

```
python ./create_simulation_config.py \
    --exp-config-template resources/template_experiment_config.json \
    --campaign-name available_item_jan_29 \
    --human-params resources/human_params.json \
    --human-config-template resources/template_human_config_avail_item.json \
    --dyn-param resources/dyn_params.json \
    --business-name "Sprout Grocery Store" \
    --model-name "gpt-3.5-turbo"\
    --out-dir out
```

### Run the simulation with the generated config
```
python ./main.py \
       out/available_item_jan_29/Sprout\ Grocery\ Store/Ready\ To\ Eat_Sprouts\ Ready\ To\ Eat\ Edamame_14\ ounces/config.2024-01-30_15-54-21.json
```

## Run multiple experiments for the product availability example
### Generate multiple configs and run multiple experiments
```
./run_multiple_times.sh \
    30 \
    available_item_jan_29 \
    resources/template_human_config_avail_item.json \
    'Sprout Grocery Store' \
    'gpt-3.5-turbo' \
    out_jan31
```

### Read the transcripts and dump them in one location
```
python ./transcript_reader.py \
    --input-dir out_jan31 \
    --output-file out_jan31.csv
```

## Run multiple experiments for the plansight calendar example
Bilal's work should make this work
### Generate multiple configs and run multiple experiments
```
./run_multiple_times.sh \
    30 \
    appointment_booking_jan22 \
    resources/template_human_config_plansight.json \
    'Plansight' \
    'gpt-3.5-turbo' \
    out_feb1
```

### Read the transcripts and dump them in one location
```
python ./transcript_reader.py \
    --input-dir out_feb1 \
    --output-file out_feb1.csv
```