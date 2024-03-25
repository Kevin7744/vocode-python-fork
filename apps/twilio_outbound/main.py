# ===================================== IMPORTS =====================================

from vocode.streaming.models.agent import ChatGPTAgentConfig, SimpleCutOffResponse
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.telephony.server.base import TelephonyServer
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request, Form, Body
from fastapi.middleware.cors import CORSMiddleware

from memory_config import config_manager
import logging
import os
import uvicorn
import re
import sys
from typing import Optional, Dict
from vocode.marrlabs.utils.logging_utils import LoggerConvIndex
from vocode.marrlabs.utils.setenv_utils import set_env

from vocode.streaming.models.agent import ChatGPTAgentConfig, FillerAudioConfig
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.telephony.server.base import TelephonyServer, TwilioCallConfig
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.utils import create_conversation_id

from memory_config import config_manager

import vocode 
from datetime import datetime
from vocode.marrlabs.utils.campaign_utils.campaign import BusinessCampaign
from pydantic import BaseModel
from vocode.marrlabs.utils.aws_utils.dynamo_s3 import DynamoDBTable
from vocode.marrlabs.utils.campaign_utils.utils import get_campaign_configs
from vocode.marrlabs.utils.campaign_utils.calendar_generator import generate_weekday_availability

import vocode.streaming.utils.worker
import vocode.streaming.transcriber.deepgram_transcriber
import vocode.streaming.streaming_conversation
import vocode.streaming.agent.base_agent
from watchtower import CloudWatchLogHandler

from vocode.streaming.telephony.constants import (
    DEFAULT_AUDIO_ENCODING,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SAMPLING_RATE
)
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig
)
# ===================================================================================

AWS_PROFILE_NAME = os.getenv("AWS_PROFILE_NAME")
CLOUDWATCH_LOGS  = os.getenv("CLOUDWATCH_LOGS")
SERVER_VERSION   = os.getenv("SERVER_VERSION")
# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger.info(f"AWS_PROFILE_NAME {AWS_PROFILE_NAME}")



logging.setLoggerClass(LoggerConvIndex)
uw_logger = logging.getLogger('vocode.streaming.utils.worker')
tw_logger = logging.getLogger('vocode.streaming.transcriber.deepgram_transcriber_profiling')
sc_logger = logging.getLogger('vocode.streaming.streaming_conversation_profiling')
lm_logger = logging.getLogger('vocode.streaming.agent.base_agent_profiling')
# restart logging state
logging.setLoggerClass(logging.Logger)

fmt = logging.Formatter('%(asctime)s|%(module)s|%(funcName)s|%(message)s')

uw_logger.setLevel(logging.INFO)
tw_logger.setLevel(logging.INFO)
sc_logger.setLevel(logging.INFO)
lm_logger.setLevel(logging.INFO)


def get_call_attributes(event):
  """
  Util to get call attributes such as campaign info
  and business info. It is expected that this function will be triggered
  by an event that contains the campaignConfig and business info. Until this
  happens automatically we feed a default event.
  """
  business_campaign = event.get('business_campaign')
  businessName      = event.get('business')['name']
  businessNumber    = event.get('business')['phone']
  businessNumber    = re.sub(r"[\(\)\-\s]", "", businessNumber)
      
  attributes = {
        "campaignId"    : business_campaign.campaign_id,
        "campaignTypeId": business_campaign.campaign_type_id,
        "campaignType"  : business_campaign.campaign_type,
        "campaignName"  : business_campaign.campaign_name,
        "businessType"  : business_campaign.business_type,
        "businessNumber": businessNumber,
        "businessName"  : businessName,
        "firstPrompt"   : business_campaign.first_prompt_for_biz(businessName),
        "llmPrompt"     : business_campaign.llm_prompt_for_biz(businessName),
        "chatbotType"   : business_campaign.campaign_type,
        "chatbotName"   : "VocodeBotV1",
        "userHistory"   : "[]",
        "agentHistory"  : "[]"
  }

  return attributes

def get_connection():
  # We need a base URL for Twilio to talk to:
  # If you're self-hosting and have an open IP/domain, 
  # set it here or in your env.
  BASE_URL = os.getenv("BASE_URL")
  # define default port
  port     = 3000
  # If neither of the above are true, we need a tunnel.
  if not BASE_URL:
    from pyngrok import ngrok
    ngrok_auth = os.environ.get("NGROK_AUTH_TOKEN")
    if ngrok_auth is not None:
      ngrok.set_auth_token(ngrok_auth)
    port = sys.argv[sys.argv.index("--port") +
                    1] if "--port" in sys.argv else 3000

    # Open a ngrok tunnel to the dev server
    BASE_URL = ngrok.connect(port).public_url.replace("https://", "")
    logger.info("ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(
      BASE_URL, port))

    logger.info(f"BASE_URL {BASE_URL}")
  return BASE_URL, port

# Initialize App 
app       = FastAPI(docs_url=None)
templates = Jinja2Templates(directory="templates")
# Set a bunch of env variables
set_env(environment="dev", aws_profile_name=AWS_PROFILE_NAME)
# First we will open up our TelephonyServer, which opens a path at
# our BASE_URL. Once we have a path, we can request a call from
# Twilio to Zoom's dial-in service or any phone number.
BASE_URL, port = get_connection()
HOST = "0.0.0.0"
# I added this because my browser blocked the request because it 
# violates the CORS policy according to the internet. 
# CORS = Cross-Origin Resource Sharing 
# CORS is a security feature that restricts web applications from 
# making requests to a domain different from the one that 
# served the web page, unless the other domain explicitly allows it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    # allow_origins=[f"http://{HOST}:{port}", f"http://127.0.0.1:{port}"],  
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Now we need a Twilio account and number from which to make our call.
# You can make an account here: 
# https://www.twilio.com/docs/iam/access-tokens#step-2-api-key
# @Roy: record = True causes the recording to show up on Twilio dashboard here:
# https://console.twilio.com/us1/monitor/logs/call-recordings
# recordings_dir is not None will cause twilio_outbound to write out 
# the human and agent wav files as well as their combined 2-channel wav. 
TWILIO_CONFIG = TwilioConfig(
  account_sid=vocode.getenv("TWILIO_ACCOUNT_SID"),
  auth_token=vocode.getenv("TWILIO_AUTH_TOKEN"),
  record=True,
  recordings_dir="recordings",
)
TWILIO_PHONE = vocode.getenv("OUTBOUND_CALLER_NUMBER")

# Store the state of the call in memory, but we can also use Redis.
# https://docs.vocode.dev/telephony#accessing-call-information-in-your-agent
#TODO: switch to storing to Redis database RedisConfigManager()
CONFIG_MANAGER = config_manager

# Get Campaign Configs
campaign_configs = get_campaign_configs("outboundCampaign-campaignConfigs", AWS_PROFILE_NAME)

# Let's create and expose that TelephonyServer.
telephony_server = TelephonyServer(
  base_url=BASE_URL,
  config_manager=CONFIG_MANAGER,
  logger=logger,
)
app.include_router(telephony_server.get_router())

# The default speech to text engine is DeepGram, so set
# the env variable DEEPGRAM_API_KEY to your Deepgram API key.
# https://deepgram.com/
# The default endpointing algorithm used by deepgram is 
# PunctuationEndpointingConfig with a 
# time_cutoff_seconds = 0.4 seconds
TRANSCRIBER_CONFIG = DeepgramTranscriberConfig(
            sampling_rate=DEFAULT_SAMPLING_RATE,
            audio_encoding=DEFAULT_AUDIO_ENCODING,
            chunk_size=DEFAULT_CHUNK_SIZE,
            model="nova-2-phonecall",
            endpointing_config=PunctuationEndpointingConfig(),
            vad_events="true",
            endpointing=10
        )
TRANSCRIBER_CONFIG.publish_audio = True

# Use StreamElements for speech synthesis here if you want fast and
# free, but there are plenty of other options that are slower but
# higher quality (like Eleven Labs below, needs key) available here:
# vocode.streaming.models.synthesizer.
#TODO: switch to synthesizer being specified in the webapp
# SYNTH_CONFIG = StreamElementsSynthesizerConfig.from_telephone_output_device()
#NATASHA_VOICE_ID = "78i7H6Q5MDhcby3e9u1w"
# NATASHA_VOICE_ID = "e78LIMbH3GBXPSrFpelg"

NATASHA_VOICE_ID = "qrQSgHxlDzi41VIGdnuM"
SYNTH_CONFIG = ElevenLabsSynthesizerConfig.from_telephone_output_device(
  api_key=vocode.getenv("ELEVEN_LABS_API_KEY"),
  similarity_boost=1,
  stability=1,
  style=1,
  use_speaker_boost=True,
  voice_id=NATASHA_VOICE_ID,
  publish_audio=True,
  experimental_streaming=True,
  optimize_streaming_latency=2
  )

def prepare_agent_config(business_campaign: BusinessCampaign, business_name: str,
                         model_name: str, allow_agent_to_be_cut_off: bool, 
                         initial_message_interruptible: bool, 
                         allowed_idle_time_seconds: int, 
                         check_in_idle_time_seconds: int, 
                         timeout_initial_message: int):
   # configure agent and its objective.
   # We'll use ChatGPT here, but you can import other models like
   # GPT4AllAgent and ChatAnthropicAgent.
   # Don't forget to set OPENAI_API_KEY!
    
    if business_campaign.business_type == "sales":
      today_date_str = datetime.today().strftime("%Y-%m-%d")
      today_date = datetime.strptime(today_date_str, "%Y-%m-%d")
      agent_availability = generate_weekday_availability(start_date=today_date, 
                                                         unavailability_odds=1)
      logger.debug(f"Agent Availability: {agent_availability}")
      llm_prompt_for_agent = business_campaign.llm_prompt_for_sales(business_name,
                                                                     str(agent_availability))
    else:
       llm_prompt_for_agent = business_campaign.llm_prompt_for_biz(business_name)
    
    AGENT_CONFIG = ChatGPTAgentConfig(
        initial_message=BaseMessage(text=business_campaign.first_prompt_for_biz(business_name)),
        prompt_preamble=llm_prompt_for_agent,
        generate_responses=True,
        end_conversation_on_goodbye=True,
        cut_off_response=SimpleCutOffResponse(),
        max_tokens=1000,
        model_name=model_name,
        allow_agent_to_be_cut_off=allow_agent_to_be_cut_off,
        initial_message_interruptible=initial_message_interruptible,
        allowed_idle_time_seconds=allowed_idle_time_seconds,
        check_in_idle_time_seconds=check_in_idle_time_seconds,
        timeout_initial_message=timeout_initial_message,
        emit_filler_if_long_response_threshold_sec = 0.1,
        send_filler_audio=FillerAudioConfig(
          silence_threshold_seconds=0.5,
          use_phrases=True,
          use_typing_noise=False
        )
    )
    return AGENT_CONFIG

def log_post_processing(input_log_file):
    with open(input_log_file, 'r+') as f:
      data=[]
      counter=0
      for l in f.readlines():
          ll = l.strip('\n').split('|')
          if ll[3] == 'speech_endpointtime':
              counter+=1
          data.append('|'.join(ll[:-1] + [str(counter)]) + '\n')
      
      f.seek(0)
      f.writelines(data)    
      
   
# OutboundCall asks Twilio to call to_phone using our Twilio phone number
# and open an audio stream to our TelephonyServer.
def _log_start_outbound_call(conversation_id, attributes):
  # Log known call details to DynamoDB as a new item before starting the outbound call.
  # TODO: add callDate, callTimestamp, and formattedTimeStamp (ISO 8601) attributes.
  new_item = dict(
    contactId = conversation_id,
    # callDate
    # callTimestamp
    # formattedTimeStamp
    campaignId = attributes["campaignId"],
    campaignName = attributes["campaignName"],
    campaignType = attributes["campaignType"],
    customerName = attributes["businessName"],
    customerPhoneNumber = attributes["businessNumber"]
  )
  dynamodb_table = DynamoDBTable('outboundCampaign-contactDetails', 
                                 aws_profile_name=AWS_PROFILE_NAME)
  dynamodb_table.create_item(new_item)

async def start_outbound_call(to_phone: Optional[str], 
                              event: Dict,
                              agent_config: Dict):
  if to_phone:
    # @Roy this is the conversation_id. Twilio also has a twilio_sid which is 
    # the call_sid but I have not gotten yet to seeing how to get it. 
    conversation_id = create_conversation_id()
    event["business"]["phone"] = to_phone

    # We can use this attributes to log them, or save in db
    attributes = get_call_attributes(event)

    # logging
    if CLOUDWATCH_LOGS and SERVER_VERSION:      
      # twilio Sid
      #cloudwatch_logs = boto3.client('logs')
      # callid
      cloudwatch_handler = CloudWatchLogHandler(log_group=f'/vocode/{SERVER_VERSION}',
                                                stream_name=f'conversation_id={conversation_id}', 
                                                create_log_group=True,
                                                use_queues=True
                                                )
      cloudwatch_handler.setFormatter(fmt)
      uw_logger.addHandler(cloudwatch_handler)
      tw_logger.addHandler(cloudwatch_handler)
      sc_logger.addHandler(cloudwatch_handler)
      lm_logger.addHandler(cloudwatch_handler)
    else:
      file_handler = logging.FileHandler(f"logs/conversation_id_{conversation_id}.log")
      file_handler.setFormatter(fmt)
      uw_logger.addHandler(file_handler)
      tw_logger.addHandler(file_handler)
      sc_logger.addHandler(file_handler)
      lm_logger.addHandler(file_handler)

    outbound_call = OutboundCall(base_url=BASE_URL,
                                 to_phone=to_phone,
                                 from_phone=TWILIO_PHONE,
                                 config_manager=CONFIG_MANAGER,
                                 agent_config=agent_config,
                                 twilio_config=TWILIO_CONFIG,
                                 transcriber_config=TRANSCRIBER_CONFIG,
                                 synthesizer_config=SYNTH_CONFIG,
                                 mobile_only=False,
                                 conversation_id=conversation_id,
                                 logger=logger)
    
    _log_start_outbound_call(conversation_id, attributes)
    await outbound_call.start()

# Expose the starter webpage
@app.get("/")
async def root(request: Request):
  env_vars = {
    "BASE_URL": BASE_URL,
    "OPENAI_API_KEY": vocode.getenv("OPENAI_API_KEY"),
    "DEEPGRAM_API_KEY": vocode.getenv("DEEPGRAM_API_KEY"),
    "TWILIO_ACCOUNT_SID": vocode.getenv("TWILIO_ACCOUNT_SID"),
    "TWILIO_AUTH_TOKEN": vocode.getenv("TWILIO_AUTH_TOKEN"),
    "OUTBOUND_CALLER_NUMBER": vocode.getenv("OUTBOUND_CALLER_NUMBER")
  }

  return templates.TemplateResponse("index.html", {
    "request": request,
    "env_vars": env_vars,
    "campaign_configs": campaign_configs
  })

@app.post("/twilio/status")
async def receive_status(request: Request):
    data = await request.form()
    data = dict(data)
    logger.info(f"Call status received {data['CallStatus']}")
    return {"message": f"Status received {data['CallStatus']}"}

class CampaignRequest(BaseModel):
    campaign_name: str
    business_name: str
    business_number: str
    dynamic_params: dict = {}
    model_name: str
    allow_agent_to_be_cut_off: bool
    initial_message_interruptible: bool
    allowed_idle_time_seconds: int
    check_in_idle_time_seconds: int
    timeout_initial_message: int

@app.post("/start_outbound_call")
async def api_start_outbound_call(request: CampaignRequest = Body(...)):
    campaign_config = campaign_configs.get(request.campaign_name)
    dynamic_params  = request.dynamic_params
    business_name   = request.business_name
    business_number   = request.business_number
    if campaign_config:
      # Update the Dynamic parameters of campaign config
      for parameter in campaign_config['campaignParams']['DynParams'].keys():
        campaign_config['campaignParams']['DynParams'][parameter]['Default'] = dynamic_params[parameter]

      # Use the selected campaign configuration to configure the call
      business_campaign = BusinessCampaign.from_dict(campaign_config)
        
      # Prepare Configuration From Input
      agent_config = prepare_agent_config(
            business_campaign, 
            business_name, 
            model_name=request.model_name, 
            allow_agent_to_be_cut_off=request.allow_agent_to_be_cut_off,
            initial_message_interruptible=request.initial_message_interruptible,
            allowed_idle_time_seconds=request.allowed_idle_time_seconds,
            check_in_idle_time_seconds=request.check_in_idle_time_seconds,
            timeout_initial_message=request.timeout_initial_message
        )
      # Construct Event
      campaign_event = {
          "business_campaign": business_campaign,
          "business": {
            "name":business_name
          }
      }
      await start_outbound_call(business_number, 
                                campaign_event,
                                agent_config)
      
      

      return {"status": "success", "config": campaign_config}
    else:
      return {"status": "error", "message": "Campaign not found"}
    
if __name__ == '__main__':
   uvicorn.run(app, host=HOST, port=port)

