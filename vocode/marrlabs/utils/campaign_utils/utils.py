from vocode.marrlabs.utils.aws_utils.dynamo_s3 import DynamoDBTable

# Parse Dynamo Response
def parse_dynamo_response(item):
  """
  Util function to parse a response if it came from dynamoDB
  Recursively converts various data types from DynamoDB format to Python-friendly structures (strings, numbers, lists, booleans, dictionaries).
  """
  if isinstance(item, dict):
      if 'S' in item:
          return item['S']
      elif 'M' in item:
          return {k: parse_dynamo_response(v) for k, v in item['M'].items()}
      elif 'N' in item:
          return int(item['N'])
      elif 'L' in item:
          return [parse_dynamo_response(v) for v in item['L']]
      elif 'BOOL' in item:
          return item['BOOL']
      else:
          return {k: parse_dynamo_response(v) for k, v in item.items()}
  elif isinstance(item, list):
      return [parse_dynamo_response(v) for v in item]
  else:
      return item
  
def get_campaign_configs(table_name, AWS_PROFILE_NAME):
    table = DynamoDBTable(table_name, aws_profile_name=AWS_PROFILE_NAME)

    response = table.scan_table()

    # Check if items are returned
    if 'Items' not in response:
        return {}

    # Organize items by campaignType
    campaign_configs = {}
    for item in response['Items']:
        item = parse_dynamo_response(item)
        campaign_type = item.get('campaignType')
        if campaign_type:
            campaign_configs[campaign_type] = item

    return campaign_configs
