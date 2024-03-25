#!/bin/bash

show_help() {
    echo "Usage: $0 [n] [arg1] [arg2] [arg3] [arg4]"
    echo "Where:"
    echo "  n               - Number of times to run the script (positive integer)"
    echo "  campaign_name"
    echo "  human_config    - Path to human config template"
    echo "  business_name"
    echo "  model_name"
    echo "  output_dir"
    echo
    echo "Example: $0 5 \\"
    echo "              'available_item_jan_29' \\"
    echo "              'template_human_config_avail_item.json' \\"
    echo "              'Sprout Grocery Store'\\"
    echo "              'gpt-3.5-turbo'\\"
    echo "              'out'"
}

# Check if help is requested
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_help
    exit 0
fi

n=$1        # number of times to run
campaign_name=$2  
human_config=$3   
business_name=$4 
model_name=$5
output_dir=$6 

# Check for correct number of arguments
if [ $# -ne 6 ]; then
    echo "Error: Incorrect number of arguments."
    show_help
    exit 1
fi
# Check if n is a number and greater than 0
if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -le 0 ]; then
    echo "Please provide a positive number as the first argument."
    show_help
    exit 1
fi

# Check if the string arguments are not empty
if [ -z "$campaign_name" ] || [ -z "$human_config" ] || [ -z "$business_name" ] || [ -z "$model_name" ] || [ -z "$output_dir" ]; then
    echo "Please provide 4 non-empty string arguments."
    show_help
    exit 1
fi

echo "String arguments: $campaign_name, $human_config, $business_name, $model_name, $output_dir"

for ((i=1; i<=n; i++))
do

   echo "Run number: $i"
   python ./create_simulation_config.py \
    --exp-config-template resources/template_experiment_config.json \
    --campaign-name "$campaign_name" \
    --human-params resources/human_params.json \
    --human-config-template "$human_config"\
    --dyn-param resources/dyn_params.json \
    --business-name "$business_name" \
    --model-name "$model_name" \
    --out-dir "$output_dir"
done

find . | grep "$output_dir" | grep json > config.list

while IFS= read -r config; do
    echo "$config"
    python ./main.py "$config"

done < config.list