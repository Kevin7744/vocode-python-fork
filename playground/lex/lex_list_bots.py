import boto3

# Assuming you've already set up a session with the correct profile
session = boto3.Session(profile_name='marrlabsmfa')
lex_client = session.client('lexv2-models')

def list_lex_bots():
    response = lex_client.list_bots()
    bots_summary = response['botSummaries']
    bots_info = []
    for bot in bots_summary:
        bot_id = bot['botId']
        bot_aliases = lex_client.list_bot_aliases(botId=bot_id)['botAliasSummaries']
        bot_aliases_names = []
        for bot_alias in bot_aliases:
            bot_aliases_names.append((bot_alias['botAliasName'],bot_alias['botAliasId']))

        bots_info.append({"botName": bot['botName'], "botId": bot_id, "botAliases":bot_aliases_names})
    return bots_info

# Example usage
bots_info = list_lex_bots()
print(bots_info)
