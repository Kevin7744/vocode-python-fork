import boto3
import json

# Define your bot configuration
bot_id = "YNJWYTSOLZ"
bot_alias_id = "LN1OTIKIQX"
locale_id = "en_US"
session_id = "testuser123"
profile_name = 'marrlabsmfa'

# Define a list of test cases
# MenuPriceIntent
# ClosedIntent
# WaitingIntent
# GreetingIntent
# OpeningHoursIntent
# FallbackIntent
test_cases = [
    {"input": "Hello there, we open at 7:00pm", "expected_intent": "OpeningHoursIntent"},
    {"input": "Good morning", "expected_intent": "GreetingIntent"},
    {"input": "Please wait so I can check", "expected_intent": "WaitingIntent"},
    {"input": "I dont know what you're talking about", "expected_intent": "FallbackIntent"},
    {"input": "We are closed now, please call back tomorrow.", "expected_intent": "ClosedIntent"},
    {"input": "Price of the pizza is $7", "expected_intent": "MenuPriceIntent"},
    # Add more test cases as needed
]

# Function to call Lex V2 bot
def call_lex_v2_bot(text_input):
    session = boto3.Session(profile_name=profile_name)
    lex_runtime = session.client('lexv2-runtime')
    response = lex_runtime.recognize_text(
        botId=bot_id,
        botAliasId=bot_alias_id,
        localeId=locale_id,
        sessionId=session_id,
        text=text_input
    )
    return response

# Function to parse Lex response and get the top intent
def get_top_intent(response):
    if 'interpretations' in response and response['interpretations']:
        top_interpretation = response['interpretations'][0]
        intent = top_interpretation.get('intent', {})
        return intent.get('name')
    return None

# Function to run test cases
def run_test_cases():
    correct = 0
    for test in test_cases:
        response = call_lex_v2_bot(test['input'])
        detected_intent = get_top_intent(response)
        if detected_intent == test['expected_intent']:
            correct += 1
            print(f"Test passed for input: '{test['input']}'. Intent: {detected_intent}")
        else:
            print(f"Test failed for input: '{test['input']}'. Expected: {test['expected_intent']}, Detected: {detected_intent}")
    print(f"Accuracy: {correct / len(test_cases) * 100}%")

# Run the test cases
run_test_cases()
