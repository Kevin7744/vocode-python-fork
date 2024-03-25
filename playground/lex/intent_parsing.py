import boto3
import json

def put_session(bot_id, bot_alias_id, locale_id, session_id, session_state, messages, profile_name='marrlabsmfa'):
    session = boto3.Session(profile_name=profile_name)
    lex_runtime = session.client('lexv2-runtime')
    response = lex_runtime.put_session(
        botId=bot_id,
        botAliasId=bot_alias_id,
        localeId=locale_id,
        sessionId=session_id,
        sessionState=session_state,
        messages=messages
    )
    print(response)
    print("-------------")
    return response

def recognize_text(bot_id, bot_alias_id, locale_id, session_id, text_input, profile_name='marrlabsmfa'):
    # Create a session and Lex V2 runtime client
    session = boto3.Session(profile_name=profile_name)
    lex_runtime = session.client('lexv2-runtime')

    # Call Lex V2 with the provided text
    response = lex_runtime.recognize_text(
        botId=bot_id,
        botAliasId=bot_alias_id,
        localeId=locale_id,
        sessionId=session_id,
        text=text_input
    )
    print(json.dumps(response, indent=4))
    print("-------------")
    return response

def parse_lex_response(response):
    if 'interpretations' in response:
        for interpretation in response['interpretations']:
            intent = interpretation.get('intent', {})
            intent_name = intent.get('name', 'No intent recognized')
            intent_state = intent.get('state', 'No state')
            slots = intent.get('slots', {})
            nlu_confidence = interpretation.get('nluConfidence', {}).get('score', 'No score')

            print(f"Intent Name: {intent_name}")
            print(f"Intent State: {intent_state}")
            print(f"NLU Confidence: {nlu_confidence}")
            print("Slots:", json.dumps(slots, indent=4))
            print("---------------------------------")



# Example usage
bot_id = "YNJWYTSOLZ"
bot_alias_id = "LN1OTIKIQX"
locale_id = "en_US"
session_id = "testuser123"
text_input = "Hello Ruth, we open at 7 at night."

response = recognize_text(bot_id, bot_alias_id, locale_id, session_id, text_input)
parse_lex_response(response)