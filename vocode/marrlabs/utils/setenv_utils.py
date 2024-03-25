import boto3
import vocode

from vocode.marrlabs.utils.aws_utils.aws_secret_manager import AwsSecretsManagerReader

def set_env(environment:str="dev", aws_profile_name="marr_labs"):
  """
  Util function to read secrets using AwsSecretsManagerReader
  """
  assert environment in ["dev", "prod"]

  # Setup default boto3 profile
  boto3.setup_default_session(profile_name=aws_profile_name, 
                              region_name="us-west-2")

  # Location Secrets in AWS
  OPENAI_SECRET_KEY                 = f"{environment}/OpenAI/api_key"
  TWILIO_SECRET_KEY                 = f"{environment}/Twilio/access_key"
  DEEPGRAM_SECRET_KEY               = f"{environment}/DeepGram/api_key"
  ELEVEN_LABS_SECRET_KEY            = f"{environment}/ElevenLabs/api_key"

  OPENAI_API_KEY_STR                = "OPENAI_API_KEY"
  DEEPGRAM_API_KEY_STR              = "DEEPGRAM_APIKEY"
  TWILIO_ACCOUNT_SID_STR            = "ACCOUNT_SID"
  TWILIO_AUTH_TOKEN_STR             = "AUTH_TOKEN"
  TWILIO_OUTBOUND_CALLER_NUMBER_STR = "OUTBOUND_CALLER_NUMBER"
  ELEVEN_LABS_API_KEY_STR           = "ELEVEN_LABS_API_KEY"

  # Get AWS Secrets
  secret_manager     = AwsSecretsManagerReader("us-west-2", 
                                               aws_profile_name=aws_profile_name)
  openai_secrets     = secret_manager.get_secret(OPENAI_SECRET_KEY)
  twilio_secrets     = secret_manager.get_secret(TWILIO_SECRET_KEY)
  deepgram_secrets   = secret_manager.get_secret(DEEPGRAM_SECRET_KEY)
  elevenlabs_secrets = secret_manager.get_secret(ELEVEN_LABS_SECRET_KEY)
  
  # Setup vocode environment
  vocode.setenv(
      OPENAI_API_KEY=openai_secrets.get(OPENAI_API_KEY_STR),
      DEEPGRAM_API_KEY=deepgram_secrets.get(DEEPGRAM_API_KEY_STR),
      TWILIO_ACCOUNT_SID=twilio_secrets.get(TWILIO_ACCOUNT_SID_STR),
      TWILIO_AUTH_TOKEN=twilio_secrets.get(TWILIO_AUTH_TOKEN_STR),
      OUTBOUND_CALLER_NUMBER=twilio_secrets.get(TWILIO_OUTBOUND_CALLER_NUMBER_STR),
      ELEVEN_LABS_API_KEY=elevenlabs_secrets.get(ELEVEN_LABS_API_KEY_STR)
  )