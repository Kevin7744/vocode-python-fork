# Define a function to show help
function Show-Help {
    Write-Host "Usage: $0 [n] [arg1] [arg2] [arg3] [arg4]"
    Write-Host "Where:"
    Write-Host "  n               - Number of times to run the script (positive integer)"
    Write-Host "  campaign_name"
    Write-Host "  human_config    - Path to human config template"
    Write-Host "  business_name"
    Write-Host "  model_name"
    Write-Host "  output_dir"
    Write-Host
    Write-Host "Example: $0 5 \"
    Write-Host "              'available_item_jan_29\'"
    Write-Host "              'template_human_config_avail_item.json\'"
    Write-Host "              'Sprout Grocery Store\'"
    Write-Host "              'gpt-3.5\'"
    Write-Host "              'out\'"
}

# Check if help is requested
if ($args[0] -eq "-h" -or $args[0] -eq "--help") {
    Show-Help
    exit
}

# Assign arguments to variables
$n = $args[0]  # number of times to run
$campaign_name = $args[1]
$human_config = $args[2]
$business_name = $args[3]
$model_name = $args[4]
$output_dir = $args[5]

# Check for correct number of arguments
if ($args.Count -ne 6) {
    Write-Host "Error: Incorrect number of arguments."
    Show-Help
    exit 1
}

# Check if n is a number and greater than 0
if (-not $n -match '^\d+$' -or $n -le 0) {
    Write-Host "Please provide a positive number as the first argument."
    Show-Help
    exit 1
}

# Check if the string arguments are not empty
if ([string]::IsNullOrWhiteSpace($campaign_name) -or [string]::IsNullOrWhiteSpace($human_config) -or [string]::IsNullOrWhiteSpace($business_name) -or [string]::IsNullOrWhiteSpace($model_name) -or [string]::IsNullOrWhiteSpace($output_dir)) {
    Write-Host "Please provide 5 non-empty string arguments."
    Show-Help
    exit 1
}

Write-Host "String arguments: $campaign_name, $human_config, $business_name, $model_name, $output_dir"

for ($i = 1; $i -le $n; $i++) {
    Write-Host "Run number: $i"
    python .\create_simulation_config.py `
        --exp-config-template resources\template_experiment_config.json `
        --campaign-name "$campaign_name" `
        --human-params resources\human_params.json `
        --human-config-template "$human_config" `
        --dyn-param resources\dyn_params.json `
        --business-name "$business_name" `
        --model-name "$model_name" `
        --out-dir "$output_dir"
}

$configList = Get-ChildItem -Recurse -File | Where-Object { $_.FullName -match "$output_dir" -and $_.Extension -eq ".json" } | Select-Object -ExpandProperty FullName
$configList | ForEach-Object {
    Write-Host "$_"
    python .\main.py "$_"
}
