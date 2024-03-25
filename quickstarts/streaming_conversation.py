import asyncio
import logging
import signal
from dotenv import load_dotenv


load_dotenv()

from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.transcriber import *
from vocode.streaming.agent import *
from vocode.streaming.synthesizer import *
from vocode.streaming.models.transcriber import *
from vocode.streaming.models.agent import *
from vocode.streaming.models.synthesizer import *
from vocode.streaming.models.message import BaseMessage
from vocode.utils.aws_utils.aws_secret_manager import AwsSecretsManagerReader

import vocode 

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def set_env():
  """
  Util function to read secrets using AwsSecretsManagerReader
  """
  OPENAI_SECRET_KEY = "dev/OpenAI/api_key"
  TWILIO_SECRET_KEY = "dev/Twilio/access_key"
  DEEPGRAM_SECRET_KEY = "dev/DeepGram/api_key"
  ELEVEN_LABS_SECRET_KEY = "dev/ElevenLabs/api_key"

  OPENAI_API_KEY_STR = "OPENAI_API_KEY"
  DEEPGRAM_API_KEY_STR = "DEEPGRAM_APIKEY"
  TWILIO_ACCOUNT_SID_STR = "ACCOUNT_SID"
  TWILIO_AUTH_TOKEN_STR = "AUTH_TOKEN"
  TWILIO_OUTBOUND_CALLER_NUMBER_STR = "OUTBOUND_CALLER_NUMBER"
  ELEVEN_LABS_API_KEY_STR = "ELEVEN_LABS_API_KEY"

  secret_manager = AwsSecretsManagerReader("us-west-2",
                                           aws_profile_name="marr_labs")
  openai_secrets = secret_manager.get_secret(OPENAI_SECRET_KEY)
  twilio_secrets = secret_manager.get_secret(TWILIO_SECRET_KEY)
  deepgram_secrets = secret_manager.get_secret(DEEPGRAM_SECRET_KEY)
  elevenlabs_secrets = secret_manager.get_secret(ELEVEN_LABS_SECRET_KEY)
  
  vocode.setenv(
      OPENAI_API_KEY=openai_secrets.get(OPENAI_API_KEY_STR),
      DEEPGRAM_API_KEY=deepgram_secrets.get(DEEPGRAM_API_KEY_STR),
      TWILIO_ACCOUNT_SID=twilio_secrets.get(TWILIO_ACCOUNT_SID_STR),
      TWILIO_AUTH_TOKEN=twilio_secrets.get(TWILIO_AUTH_TOKEN_STR),
      OUTBOUND_CALLER_NUMBER=twilio_secrets.get(TWILIO_OUTBOUND_CALLER_NUMBER_STR),
      ELEVEN_LABS_API_KEY=elevenlabs_secrets.get(ELEVEN_LABS_API_KEY_STR)
  )

async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=True,
        logger=logger,
        use_blocking_speaker_output=True,  # this moves the playback to a separate thread, set to False to use the main thread
    )

    set_env()
    NATASHA_VOICE_ID = "78i7H6Q5MDhcby3e9u1w"

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(
            DeepgramTranscriberConfig.from_input_device(
                microphone_input,
                endpointing_config=PunctuationEndpointingConfig(),
            )
        ),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                send_filler_audio=False,
                generate_responses=True,
                model_name="gpt-4-1106-preview",
                initial_message=BaseMessage(
                    text="Hey, thank you for calling customer service. This conversation will be recorded. How can I help you?"
                ),
                prompt_preamble="""You are helpful virtual agent that handles customer service conversations over the phone.
                You are a considerate and respectful virtual assistant designed to assist with customer service inquiries over the telephone. At the beginning of each interaction, I would like you to inform customers that their conversation will be recorded for quality assurance purposes. It's essential to maintain a friendly tone throughout every exchange while using concise language, keeping responses succinct yet engaging. This approach ensures compatibility with telephony systems utilizing mu-law encoding and speech-to-text technology.

                Throughout your interactions, avoid providing specific advice on legal, financial, or medical matters, instead directing users towards seeking professional guidance in those areas. 
                When faced with complex queries beyond your knowledge base, kindly express your limitations without causing frustration. 
                Politely gather contact details to follow up via callback or email when further assistance becomes available.
                
                Here are some guidelines to help navigate common scenarios:
                
                If asked for personal or identifying information, gently decline by emphasizing data privacy regulations and focus on resolving immediate concerns within the current scope.
                Refrain from discussing internal operations, policies, or procedures unless they directly contribute to addressing the customer's query. Maintaining confidentiality is crucial.
                Should a user become upset or angry during the conversation, remain calm and patient, validating their feelings before attempting to refocus the discussion on finding solutions together.
                If someone requests access to proprietary resources or sensitive material unrelated to their concern, diplomatically explain why such sharing isn't feasible at this time.
                In case a caller insists on speaking with a human representative immediately, apologize for any inconvenience caused and assure them that you will escalate their request appropriately after collecting necessary contact information.
                Be prepared to handle situations where users may ask about the nature of your existence; simply state that you are an advanced AI system designed to provide exceptional customer support. Avoid revealing technical aspects or intricacies related to your programming or operation.
                Lastly, always thank the customer for their patience and understanding, regardless of whether their issue has been fully resolved during the initial interaction.
                """,
            )
        ),
        synthesizer=ElevenLabsSynthesizer(ElevenLabsSynthesizerConfig.from_output_device(
            speaker_output,
            api_key=vocode.getenv("ELEVEN_LABS_API_KEY"),
            similarity_boost=1,
            stability=1,
            style=1,
            use_speaker_boost=True,
            voice_id=NATASHA_VOICE_ID
            )
        ),
        logger=logger,
    )
    await conversation.start()
    print("Conversation started, press Ctrl+C to end")
    signal.signal(
        signal.SIGINT, lambda _0, _1: asyncio.create_task(conversation.terminate())
    )
    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
