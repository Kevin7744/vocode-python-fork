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
from vocode.streaming.pubsub.base_pubsub import *
from vocode import pubsub
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
    )
    set_env()
    transcriber_config = DeepgramTranscriberConfig.from_input_device(
        microphone_input,
        endpointing_config=PunctuationEndpointingConfig(),
        publish_audio=True,
    )

    audio_recorder = AudioFileWriterSubscriber(
        "streaming_conversation",
        save_chunk_in_sec=1,
        sampling_rate=transcriber_config.sampling_rate,
    )

    pubsub.subscribe(audio_recorder, PubSubTopics.INPUT_AUDIO_STREAMS)
    audio_recorder.start()

    NATASHA_VOICE_ID = "78i7H6Q5MDhcby3e9u1w"

    SYNTH_CONFIG = ElevenLabsSynthesizerConfig.from_output_device(
        speaker_output,
    api_key=vocode.getenv("ELEVEN_LABS_API_KEY"),
    similarity_boost=1,
    stability=1,
    style=1,
    use_speaker_boost=True,
    voice_id=NATASHA_VOICE_ID
    )

    conversation = StreamingConversation(
        output_device=speaker_output,
        transcriber=DeepgramTranscriber(transcriber_config),
        agent=ChatGPTAgent(
            ChatGPTAgentConfig(
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="""The AI is having a pleasant conversation about life. Keep your responses short. 
                Use simple and accessible English.""",
            )
        ),
        synthesizer=ElevenLabsSynthesizer(SYNTH_CONFIG)
    )

    await conversation.start()
    print("Conversation started, press Ctrl+C to end")

    def sigint_handler(signum, frame):
        asyncio.create_task(conversation.terminate())
        pubsub.stop()

    signal.signal(signal.SIGINT, sigint_handler)

    while conversation.is_active():
        chunk = await microphone_input.get_audio()
        conversation.receive_audio(chunk)


if __name__ == "__main__":
    asyncio.run(main())
