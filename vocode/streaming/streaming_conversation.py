from __future__ import annotations
import json
import asyncio
import queue
import random
import threading
from typing import Any, Awaitable, Callable, Generic, Optional, Tuple, TypeVar, cast
import logging
import time
import typing
import os
from datetime import datetime 
from pydub import AudioSegment

from vocode.streaming.action.worker import ActionsWorker

from vocode.streaming.agent.bot_sentiment_analyser import (
    BotSentimentAnalyser,
)

from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.models.actions import ActionInput
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    Message,
    Transcript,
    TranscriptCompleteEvent,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.transcriber import EndpointingConfig, TranscriberConfig
from vocode.streaming.output_device.base_output_device import BaseOutputDevice
from vocode.streaming.utils.conversation_logger_adapter import wrap_logger
from vocode.streaming.utils.events_manager import EventsManager
from vocode.streaming.utils.goodbye_model import GoodbyeModel

from vocode.streaming.models.agent import ChatGPTAgentConfig, FillerAudioConfig
from vocode.streaming.models.synthesizer import (
    SentimentConfig,
)
from vocode.streaming.constants import (
    TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS,
    PER_CHUNK_ALLOWANCE_SECONDS,
    ALLOWED_IDLE_TIME,
    CHECK_IN_IDLE_TIME,
    AGENT_RECORDING_SUFFIX,
    HUMAN_RECORDING_SUFFIX,
    ALL_RECORDING_SUFFIX
    
)
from vocode.streaming.agent.base_agent import (
    AgentInput,
    AgentResponse,
    AgentResponseFillerAudio,
    AgentResponseMessage,
    AgentResponseStop,
    AgentResponseType,
    BaseAgent,
    TranscriptionAgentInput,
)
from vocode.streaming.synthesizer.base_synthesizer import (
    BaseSynthesizer,
    SynthesisResult,
    FillerAudio,
)
from vocode.streaming.utils import create_conversation_id, get_chunk_size_per_second
from vocode.streaming.transcriber.base_transcriber import (
    Transcription,
    BaseTranscriber,
)
from vocode.streaming.utils.state_manager import ConversationStateManager
from vocode.streaming.utils.worker import (
    AsyncQueueWorker,
    InterruptibleAgentResponseWorker,
    InterruptibleEvent,
    InterruptibleEventFactory,
    InterruptibleAgentResponseEvent,
    InterruptibleWorker,
)

from vocode.streaming.pubsub.base_pubsub import AudioFileWriterSubscriber, PubSubTopics
from vocode import pubsub

from vocode.marrlabs.utils.aws_utils.dynamo_s3 import DynamoDBTable
from vocode.marrlabs.utils.logging_utils import LoggerConvIndex


OutputDeviceType = TypeVar("OutputDeviceType", bound=BaseOutputDevice)
logging.setLoggerClass(LoggerConvIndex)
sc_logger = logging.getLogger(__name__+'_profiling')
logging.setLoggerClass(logging.Logger)

#'speech_start_time': user turn first word start
#'speech_end_time': user turn last word end
#   *we know the time t_i at which the orchestrator sent the audio byte to the ASR. 
#   *when deepgram outputs the transcription, it will have a timestamp for the endpoint of speech, call it T
#   *to get the endpoint time relative to orchestrator, do t_i + T. 
#'asr_time_to_first_token': time at which orchestrator receives first part of asr transcription
#'asr_end_time': time at which orchestrator receives last part of asr transcription including the <end of speech> tag
#'gpt_start_time': time at which orchestrator sends trans to agent (LLM)
#'gpt_first_token_time': time at which orchestrator receives first text from LLM
#'gpt_last_token_time': time at which orchestrator receives last text from LLM
#'tts_start_time': time at which orchestrator sends text to synthesizer
#'tts_first_byte_time': time at which orchestrator receives first audio from synthesizer
#'tts_last_byte_time': time at which orchestrator receives the last audio from synthesizer
#'agent_audio_start_time': time at which orchestrator plays the audio. 

conversation_tags: list  = [
    'speech_start_time',
    'speech_end_time',
    'asr_time_to_first_token',
    'asr_end_time',
    'gpt_start_time',
    'gpt_first_token_time',
    'gpt_last_token_time',
    'tts_start_time',
    'tts_first_byte_time',
    'tts_last_byte_time',
    'agent_audio_start_time'
    ]

conversation_model: dict = { tag : None for tag in conversation_tags }

class StreamingConversation(Generic[OutputDeviceType]):
    class QueueingInterruptibleEventFactory(InterruptibleEventFactory):
        def __init__(self, conversation: "StreamingConversation"):
            self.conversation = conversation

        def create_interruptible_event(
            self, payload: Any, is_interruptible: bool = True
        ) -> InterruptibleEvent[Any]:
            interruptible_event: InterruptibleEvent = (
                super().create_interruptible_event(payload, is_interruptible)
            )
            self.conversation.interruptible_events.put_nowait(interruptible_event)
            return interruptible_event

        def create_interruptible_agent_response_event(
            self,
            payload: Any,
            is_interruptible: bool = True,
            agent_response_tracker: Optional[asyncio.Event] = None,
        ) -> InterruptibleAgentResponseEvent:
            interruptible_event = super().create_interruptible_agent_response_event(
                payload,
                is_interruptible=is_interruptible,
                agent_response_tracker=agent_response_tracker,
            )
            self.conversation.interruptible_events.put_nowait(interruptible_event)
            return interruptible_event

    class TranscriptionsWorker(AsyncQueueWorker):
        """Processes all transcriptions: sends an interrupt if needed
        and sends final transcriptions to the output queue"""
 
        def __init__(
            self,
            input_queue: asyncio.Queue[Transcription],
            output_queue: asyncio.Queue[InterruptibleEvent[AgentInput]],
            conversation: "StreamingConversation",
            interruptible_event_factory: InterruptibleEventFactory,
        ):
            super().__init__(input_queue, output_queue)
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.conversation = conversation
            self.interruptible_event_factory = interruptible_event_factory

        async def process(self, transcription: Transcription):
            self.conversation.mark_last_action_timestamp()

            if transcription.message.strip() == "":
                self.conversation.logger.info("Ignoring empty transcription")
                return
            
            if transcription.is_final:
                self.conversation.logger.debug(
                    "Got transcription: {}, confidence: {}".format(
                        transcription.message, transcription.confidence
                    )
                )
            self.conversation.logger.debug(f"INSIDE TRANS WORKER - process - {transcription.message}, {transcription.is_final}")
            self.conversation.logger.debug(f"self.conversation.is_human_speaking {self.conversation.is_human_speaking}")
            if (
                # this means: if the previous transcription was final
                not self.conversation.is_human_speaking
                and self.conversation.is_interrupt(transcription)
            ):
                sc_logger.info(f'interrupt_start_time|{LoggerConvIndex.conversation_relative(time.time())}|{LoggerConvIndex.conversation_idx()}')
                self.conversation.current_transcription_is_interrupt = (
                    self.conversation.broadcast_interrupt()
                )
                if self.conversation.current_transcription_is_interrupt:
                    self.conversation.logger.debug("sending interrupt")
                self.conversation.logger.debug("Human started speaking")

            transcription.is_interrupt = (
                self.conversation.current_transcription_is_interrupt
            )
            self.conversation.is_human_speaking = not transcription.is_final

            if transcription.is_final:
                # we use getattr here to avoid the dependency cycle between VonageCall and StreamingConversation
                event = self.interruptible_event_factory.create_interruptible_event(
                    TranscriptionAgentInput(
                        transcription=transcription,
                        conversation_id=self.conversation.id,
                        vonage_uuid=getattr(self.conversation, "vonage_uuid", None),
                        twilio_sid=getattr(self.conversation, "twilio_sid", None),
                    )
                )
                
                self.output_queue.put_nowait(event)

    class FillerAudioWorker(InterruptibleAgentResponseWorker):
        """
        - Waits for a configured number of seconds and then sends filler audio to the output
        - Exposes wait_for_filler_audio_to_finish() which the AgentResponsesWorker waits on before
          sending responses to the output queue
        """

        def __init__(
            self,
            input_queue: asyncio.Queue[InterruptibleAgentResponseEvent[FillerAudio]],
            conversation: "StreamingConversation",
        ):
            super().__init__(input_queue=input_queue)
            self.input_queue = input_queue
            self.conversation = conversation
            self.current_filler_seconds_per_chunk: Optional[int] = None
            self.filler_audio_started_event: Optional[threading.Event] = None

        async def wait_for_filler_audio_to_finish(self):
            if (
                self.filler_audio_started_event is None
                or not self.filler_audio_started_event.set()
            ):
                self.conversation.logger.debug(
                    "Not waiting for filler audio to finish since we didn't send any chunks"
                )
                return
            if self.interruptible_event and isinstance(
                self.interruptible_event, InterruptibleAgentResponseEvent
            ):
                await self.interruptible_event.agent_response_tracker.wait()

        def interrupt_current_filler_audio(self):
            return self.interruptible_event and self.interruptible_event.interrupt()

        async def process(self, item: InterruptibleAgentResponseEvent[FillerAudio]):
            try:
                filler_audio = item.payload
                assert self.conversation.filler_audio_config is not None
                filler_synthesis_result = filler_audio.create_synthesis_result()                
                self.current_filler_seconds_per_chunk = filler_audio.seconds_per_chunk
                silence_threshold = (
                    self.conversation.filler_audio_config.silence_threshold_seconds
                )
                await asyncio.sleep(silence_threshold)
                self.conversation.logger.debug("Sending filler audio to output")
                self.filler_audio_started_event = threading.Event()
                await self.conversation.send_speech_to_output(
                    filler_audio.message.text,
                    filler_synthesis_result,
                    item.interruption_event,
                    filler_audio.seconds_per_chunk,
                    started_event=self.filler_audio_started_event,
                    is_filler_audio=True
                )
                self.conversation.mark_last_action_timestamp()
                item.agent_response_tracker.set()
            except asyncio.CancelledError:
                pass

    class AgentResponsesWorker(InterruptibleAgentResponseWorker):
        """Runs Synthesizer.create_speech and sends the SynthesisResult to the output queue"""

        def __init__(
            self,
            input_queue: asyncio.Queue[InterruptibleAgentResponseEvent[AgentResponse]],
            output_queue: asyncio.Queue[
                InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
            ],
            conversation: "StreamingConversation",
            interruptible_event_factory: InterruptibleEventFactory,
        ):
            super().__init__(
                input_queue=input_queue,
                output_queue=output_queue,
            )
            self.input_queue = input_queue
            self.output_queue = output_queue
            self.conversation = conversation
            self.interruptible_event_factory = interruptible_event_factory
            self.chunk_size = (
                get_chunk_size_per_second(
                    self.conversation.synthesizer.get_synthesizer_config().audio_encoding,
                    self.conversation.synthesizer.get_synthesizer_config().sampling_rate,
                )
                * TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS
            )

        def send_filler_audio(self, agent_response_tracker: Optional[asyncio.Event]):
            assert self.conversation.filler_audio_worker is not None
            self.conversation.logger.debug("Sending filler audio")
            if self.conversation.synthesizer.filler_audios:
                filler_audio = random.choice(
                    self.conversation.synthesizer.filler_audios
                )
                event = self.interruptible_event_factory.create_interruptible_agent_response_event(
                    filler_audio,
                    is_interruptible=filler_audio.is_interruptible,
                    agent_response_tracker=agent_response_tracker,
                )
                self.conversation.filler_audio_worker.consume_nonblocking(event)
            else:
                self.conversation.logger.debug(
                    "No filler audio available for synthesizer"
                )

        async def process(self, item: InterruptibleAgentResponseEvent[AgentResponse]):
            if not self.conversation.synthesis_enabled:
                self.conversation.logger.debug(
                    "Synthesis disabled, not synthesizing speech"
                )
                return
            try:
                agent_response = item.payload
                if isinstance(agent_response, AgentResponseFillerAudio):
                    if not self.conversation.is_audio_playing:
                        self.send_filler_audio(item.agent_response_tracker)
                        return
                    else:
                        return
                if isinstance(agent_response, AgentResponseStop):
                    # HERE REASON OF TERMINATION:
                    # Could be unuseful conversation
                    # CloseIntent
                    # Intent is fulfilled
                    # > We need to add this intent to the AgentResponseStop
                    self.conversation.logger.debug("Agent requested to stop")
                    item.agent_response_tracker.set()
                    await self.conversation.terminate()
                    return
                
                agent_response_message = typing.cast(
                    AgentResponseMessage, agent_response
                )
                sc_logger.info(f'tts_message:{agent_response_message.message}|{LoggerConvIndex.conversation_idx()}')

                if self.conversation.filler_audio_worker is not None:
                    if (
                        self.conversation.filler_audio_worker.interrupt_current_filler_audio()
                    ):
                        await self.conversation.filler_audio_worker.wait_for_filler_audio_to_finish()

                self.conversation.logger.debug("Synthesizing speech for message")
                synthesis_result = None
                # @ Bilal we need to understand how the synthesizer is chunking the data and 
                # try to reduce that size to see if we can control the interrupt better. 
                if not agent_response_message.message.text is None:
                    sc_logger.info(f'tts_start_time|{LoggerConvIndex.conversation_relative(time.time())}|{LoggerConvIndex.conversation_idx()}') 
                    synthesis_result = await self.conversation.synthesizer.create_speech(
                        agent_response_message.message,
                        self.chunk_size,
                        bot_sentiment=self.conversation.bot_sentiment,
                    )

                sc_logger.info(f'tts_first_byte_time|{LoggerConvIndex.conversation_relative(time.time())}|{LoggerConvIndex.conversation_idx()}')                       
                self.produce_interruptible_agent_response_event_nonblocking(
                    (agent_response_message.message, synthesis_result),
                    is_interruptible=item.is_interruptible,
                    agent_response_tracker=item.agent_response_tracker,
                )
            except asyncio.CancelledError:
                pass

    class SynthesisResultsWorker(InterruptibleAgentResponseWorker):
        """Plays SynthesisResults from the output queue on the output device"""

        def __init__(
            self,
            input_queue: asyncio.Queue[
                InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
            ],
            conversation: "StreamingConversation",
        ):
            super().__init__(input_queue=input_queue)
            self.input_queue = input_queue
            self.conversation = conversation

        async def process(
            self,
            item: InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]],
        ):
            try:
                
                message, synthesis_result = item.payload
                # create an empty transcript message and attach it to the transcript
                transcript_message = Message(
                    text="",
                    sender=Sender.BOT,
                )
                self.conversation.transcript.add_message(
                    message=transcript_message,
                    conversation_id=self.conversation.id,
                    publish_to_events_manager=False,
                )
                message_sent = None
                if synthesis_result:
                    message_sent, cut_off = await self.conversation.send_speech_to_output(
                        message.text,
                        synthesis_result,
                        item.interruption_event,
                        TEXT_TO_SPEECH_CHUNK_SIZE_SECONDS,
                        transcript_message=transcript_message,
                    )
                    self.conversation.mark_last_action_timestamp()
                    self.conversation.logger.debug("Adding transcript message to Transcript")
                    self.conversation.logger.debug(f"transcript message is {transcript_message}")
                    self.conversation.logger.debug(f"transcript message text is {message.text}")
                    # publish the transcript message now that it includes what was said during send_speech_to_output
                    self.conversation.transcript.maybe_publish_transcript_event_from_message(
                        message=transcript_message,
                        conversation_id=self.conversation.id,
                    )
                
                if not message.intent is None:
                    self.conversation.logger.debug("Adding intent to Transcript")
                    self.conversation.logger.debug(f"intent is {message.intent}")
                    # publish the transcript message now that it includes what was said during send_speech_to_output
                    self.conversation.transcript.add_message(
                        message=Message(
                                    text=message.intent,
                                    sender=Sender.BOT,
                                ),
                        conversation_id=self.conversation.id,
                        publish_to_events_manager=True,
                    )
                
                item.agent_response_tracker.set()
                if synthesis_result:
                    self.conversation.logger.debug("Message sent: {}".format(message_sent))
                    if cut_off:
                        self.conversation.agent.update_last_bot_message_on_cut_off(
                            message_sent
                        )
                    
                    if self.conversation.agent.agent_config.end_conversation_on_goodbye:
                        goodbye_detected_task = (
                            self.conversation.agent.create_goodbye_detection_task(
                                message_sent
                            )
                        )
                        try:
                            if await asyncio.wait_for(goodbye_detected_task, 0.1):
                                self.conversation.logger.debug(
                                    "Agent said goodbye, ending call"
                                )
                                await self.conversation.terminate()
                        except asyncio.TimeoutError:
                            pass
            except asyncio.CancelledError:
                pass
        
    
    def __init__(
        self,
        output_device: OutputDeviceType,
        transcriber: BaseTranscriber[TranscriberConfig],
        agent: BaseAgent,
        synthesizer: BaseSynthesizer,
        conversation_id: Optional[str] = None,
        recordings_dir: Optional[str] = None,
        per_chunk_allowance_seconds: float = PER_CHUNK_ALLOWANCE_SECONDS,
        events_manager: Optional[EventsManager] = None,
        logger: Optional[logging.Logger] = None,

    ):
        self.AWS_PROFILE_NAME = os.getenv("AWS_PROFILE_NAME")
        self.recordings_dir = recordings_dir
        self.is_audio_playing = False
        self.id = conversation_id or create_conversation_id()
        
        self.logger = wrap_logger(
            logger or logging.getLogger(__name__),
            conversation_id=self.id,
        )

        self.output_device = output_device
        self.transcriber = transcriber
        self.agent = agent
        self.synthesizer = synthesizer
        self.synthesizer.set_audio_id(self.transcriber.transcription_audio_id)
        self.synthesis_enabled = True
        
        self.interruptible_events: queue.Queue[InterruptibleEvent] = queue.Queue()
        
        self.interruptible_event_factory = self.QueueingInterruptibleEventFactory(
            conversation=self
        )

        self.agent.set_interruptible_event_factory(self.interruptible_event_factory)
        
        self.synthesis_results_queue: asyncio.Queue[
            InterruptibleAgentResponseEvent[Tuple[BaseMessage, SynthesisResult]]
        ] = asyncio.Queue()
        
        self.filler_audio_queue: asyncio.Queue[
            InterruptibleAgentResponseEvent[FillerAudio]
        ] = asyncio.Queue()
        
        self.state_manager = self.create_state_manager()
        
        self.transcriptions_worker = self.TranscriptionsWorker(
            input_queue=self.transcriber.output_queue,
            output_queue=self.agent.get_input_queue(),
            conversation=self,
            interruptible_event_factory=self.interruptible_event_factory,
        )
        self.agent.attach_conversation_state_manager(self.state_manager)
        self.agent_responses_worker = self.AgentResponsesWorker(
            input_queue=self.agent.get_output_queue(),
            output_queue=self.synthesis_results_queue,
            conversation=self,
            interruptible_event_factory=self.interruptible_event_factory,
        )
        self.actions_worker = None
        if self.agent.get_agent_config().actions:
            self.actions_worker = ActionsWorker(
                input_queue=self.agent.actions_queue,
                output_queue=self.agent.get_input_queue(),
                interruptible_event_factory=self.interruptible_event_factory,
                action_factory=self.agent.action_factory,
            )
            self.actions_worker.attach_conversation_state_manager(self.state_manager)
        self.synthesis_results_worker = self.SynthesisResultsWorker(
            input_queue=self.synthesis_results_queue, conversation=self
        )
        self.filler_audio_worker = None
        self.filler_audio_config: Optional[FillerAudioConfig] = None
        if (
            self.agent.get_agent_config().send_filler_audio
            or self.agent.get_agent_config().emit_filler_if_long_response
        ):
            self.filler_audio_worker = self.FillerAudioWorker(
                input_queue=self.filler_audio_queue, conversation=self
            )

        self.events_manager = events_manager or EventsManager()
        self.events_task: Optional[asyncio.Task] = None
        self.per_chunk_allowance_seconds = per_chunk_allowance_seconds
        self.transcript = Transcript()
        self.transcript.attach_events_manager(self.events_manager)
        self.bot_sentiment = None
        if self.agent.get_agent_config().track_bot_sentiment:
            self.sentiment_config = (
                self.synthesizer.get_synthesizer_config().sentiment_config
            )
            if not self.sentiment_config:
                self.sentiment_config = SentimentConfig()
            self.bot_sentiment_analyser = BotSentimentAnalyser(
                emotions=self.sentiment_config.emotions
            )

        self.is_human_speaking = False
        self.active = False
        self.mark_last_action_timestamp()

        self.check_for_idle_task: Optional[asyncio.Task] = None
        self.track_bot_sentiment_task: Optional[asyncio.Task] = None

        self.current_transcription_is_interrupt: bool = False

        # tracing
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

        if self.recordings_dir is not None:
            self.audio_recorder = AudioFileWriterSubscriber(
                "streaming_conversation",
                save_chunk_in_sec=0.0,
                sampling_rate=self.transcriber.transcriber_config.sampling_rate,
                directory=self.recordings_dir,
            )

            pubsub.subscribe(self.audio_recorder, PubSubTopics.INPUT_AUDIO_STREAMS)
        self.timestamp = datetime.utcnow().timestamp()
        self.first_time = True
        self.first_audio_chunk = False 
        
    def create_state_manager(self) -> ConversationStateManager:
        return ConversationStateManager(conversation=self)
    
    async def start(self, mark_ready: Optional[Callable[[], Awaitable[None]]] = None):
        self.transcriber.start()
        self.transcriptions_worker.start()
        self.agent_responses_worker.start()
        self.synthesis_results_worker.start()
        self.output_device.start()
        

        if self.filler_audio_worker is not None:
            self.filler_audio_worker.start()
        if self.actions_worker is not None:
            self.actions_worker.start()
        is_ready = await self.transcriber.ready()
        if not is_ready:
            raise Exception("Transcriber startup failed")
        if (
            self.agent.get_agent_config().send_filler_audio
            or self.agent.get_agent_config().emit_filler_if_long_response
        ):
            if not isinstance(
                self.agent.get_agent_config().send_filler_audio, FillerAudioConfig
            ):
                self.filler_audio_config = FillerAudioConfig()
            else:
                self.filler_audio_config = typing.cast(
                    FillerAudioConfig, self.agent.get_agent_config().send_filler_audio
                )
            await self.synthesizer.set_filler_audios(self.filler_audio_config)

        if self.recordings_dir is not None:
            self.audio_recorder.start()

        self.agent.start()
        self.agent.attach_transcript(self.transcript)
        if mark_ready:
            await mark_ready()
        if self.synthesizer.get_synthesizer_config().sentiment_config:
            await self.update_bot_sentiment()

        self.active = True
        if self.synthesizer.get_synthesizer_config().sentiment_config:
            self.track_bot_sentiment_task = asyncio.create_task(
                self.track_bot_sentiment()
            )
        self.check_in_count      = 0
        if len(self.events_manager.subscriptions) > 0:
            self.events_task = asyncio.create_task(self.events_manager.start())
        self.start_timestamp = time.time()
        self.mark_last_action_timestamp()
        self.initial_step_complete = False
        self.transcriber_started_progress = False
        self.initial_step_task   = asyncio.create_task(self.initial_step())
        self.check_for_idle_task = asyncio.create_task(self.check_for_idle())

    async def initial_step(self):
        self.logger.debug(f"initial_step: active")
        self.logger.debug(f"active: {self.is_active()}, is_first_response: {self.agent.is_first_response}")
        while self.is_active() and self.agent.is_first_response:
            timeout = self.agent.agent_config.timeout_initial_message
            
            self.transcriber_started_progress = self.transcriber_started_progress or self.transcriber.is_in_progress()
            if time.time() - self.start_timestamp > timeout:
                self.logger.debug(f"Timeout reached {timeout}")
                if not self.transcriber_started_progress:
                    self.logger.debug(f"Initial human speech was not detected")
                    self.agent.is_first_response = False
                    initial_response = self.agent.get_agent_config().initial_message
                    self.logger.debug(f"initial_step: Sending initial message {initial_response}")
                    if initial_response:
                        self.mark_last_action_timestamp()
                        asyncio.create_task(self.send_initial_message(initial_response))
                        
                        
            await asyncio.sleep(0.15)

    async def send_initial_message(self, initial_response: BaseMessage):
        message_is_interruptible = self.agent.agent_config.initial_message_interruptible
        if not message_is_interruptible:
            self.transcriber.mute()

        # TODO: @Bilal we need to break this message up using collate_response_async
        # in utils. 
        initial_message_tracker = asyncio.Event()
        agent_response_event = (
            self.interruptible_event_factory.create_interruptible_agent_response_event(
                AgentResponseMessage(message=initial_response, 
                                     is_interruptible=message_is_interruptible),
                is_interruptible=message_is_interruptible,
                agent_response_tracker=initial_message_tracker,
            )
        )
        self.agent_responses_worker.consume_nonblocking(agent_response_event)
        await initial_message_tracker.wait()
        self.transcriber.unmute()

        return True
    
    async def check_for_idle(self):
        """
        Idle Actions Logic:
            - If we reach 5 seconds idle, send an AgentResponse.
            - If we reach 15 seconds idle, terminate the call.
        """
        while self.is_active():
            if time.time() - self.last_action_timestamp > (
                self.agent.get_agent_config().allowed_idle_time_seconds
                or ALLOWED_IDLE_TIME
            ):
                self.logger.debug("Conversation idle for too long, terminating")
                await self.terminate()
                return
    
            if time.time() - self.last_action_timestamp > (
                self.agent.get_agent_config().check_in_idle_time_seconds
                or CHECK_IN_IDLE_TIME
            ):
                if self.check_in_count > 2:
                    self.logger.debug("Reached max check in times, terminating.")
                    await self.terminate()
                    return
                self.logger.debug(f"No reply received, checking in again. {self.check_in_count}")
                self.mark_last_action_timestamp()
                message_tracker = asyncio.Event()
                messages = ["Are you still there?", "Hello ... anybody there?"]
                message  = random.choice(messages)
                agent_response_event = (
                    self.interruptible_event_factory.create_interruptible_agent_response_event(
                        AgentResponseMessage(message=BaseMessage(text=message), 
                                            is_interruptible=True),
                        is_interruptible=True,
                        agent_response_tracker=message_tracker,
                    )
                )
                self.agent_responses_worker.consume_nonblocking(agent_response_event)
                await message_tracker.wait()
                self.check_in_count += 1
            await asyncio.sleep(0.1)

    async def track_bot_sentiment(self):
        """Updates self.bot_sentiment every second based on the current transcript"""
        prev_transcript = None
        while self.is_active():
            await asyncio.sleep(1)
            if self.transcript.to_string() != prev_transcript:
                await self.update_bot_sentiment()
                prev_transcript = self.transcript.to_string()

    async def update_bot_sentiment(self):
        new_bot_sentiment = await self.bot_sentiment_analyser.analyse(
            self.transcript.to_string()
        )
        if new_bot_sentiment.emotion:
            self.logger.debug("Bot sentiment: %s", new_bot_sentiment)
            self.bot_sentiment = new_bot_sentiment

    def receive_message(self, message: str):
        transcription = Transcription(
            message=message,
            confidence=1.0,
            is_final=True,
        )
        self.transcriptions_worker.consume_nonblocking(transcription)

    def receive_audio(self, chunk: bytes):
        if not self.first_audio_chunk:
            LoggerConvIndex.set_conversation_start_time(time.time())
            self.first_audio_chunk = True;
        self.transcriber.send_audio(chunk)

    def send_synthesizer_audio(self, chunk: bytes):
        self.synthesizer.send_audio(chunk)

    def warmup_synthesizer(self):
        self.synthesizer.ready_synthesizer()

    def mark_last_action_timestamp(self):
        self.last_action_timestamp = time.time()

    def broadcast_interrupt(self):
        """Stops all inflight events and cancels all workers that are sending output

        Returns true if any events were interrupted - which is used as a flag for the agent (is_interrupt)
        """
        num_interrupts = 0
        while True:
            try:
                interruptible_event = self.interruptible_events.get_nowait()
                if not interruptible_event.is_interrupted():
                    if interruptible_event.interrupt():
                        self.logger.debug("Interrupting event")
                        num_interrupts += 1
            except queue.Empty:
                break
        self.agent.cancel_current_task()
        self.agent_responses_worker.cancel_current_task()
        return num_interrupts > 0

    def is_interrupt(self, transcription: Transcription):
        return transcription.confidence >= (
            self.transcriber.get_transcriber_config().min_interrupt_confidence or 0
        )

    async def send_speech_to_output(
        self,
        message: str,
        synthesis_result: SynthesisResult,
        stop_event: threading.Event,
        seconds_per_chunk: int,
        transcript_message: Optional[Message] = None,
        started_event: Optional[threading.Event] = None,
        is_filler_audio: Optional[bool] = False
    ):
        """
        - Sends the speech chunk by chunk to the output device
          - update the transcript message as chunks come in (transcript_message is always provided for non filler audio utterances)
        - If the stop_event is set, the output is stopped
        - Sets started_event when the first chunk is sent

        Importantly, we rate limit the chunks sent to the output. For interrupts to work properly,
        the next chunk of audio can only be sent after the last chunk is played, so we send
        a chunk of x seconds only after x seconds have passed since the last chunk was sent.

        Returns the message that was sent up to, and a flag if the message was cut off
        """
        self.is_audio_playing = True
        try:
            if self.transcriber.get_transcriber_config().mute_during_speech:
                self.logger.debug("Muting transcriber")
                self.transcriber.mute()
            message_sent = message
            cut_off = False
            chunk_size = seconds_per_chunk * get_chunk_size_per_second(
                self.synthesizer.get_synthesizer_config().audio_encoding,
                self.synthesizer.get_synthesizer_config().sampling_rate,
            )
            chunk_idx = 0
            seconds_spoken = 0

            async for chunk_result in synthesis_result.chunk_generator:
                start_time = time.time()
                speech_length_seconds = seconds_per_chunk * (
                    len(chunk_result.chunk) / chunk_size
                )
                self.logger.debug(f"speech_length_seconds {speech_length_seconds}")
                seconds_spoken = chunk_idx * seconds_per_chunk
                if stop_event.is_set():
                    sc_logger.info(f'interrupt_tts_time|{LoggerConvIndex.conversation_relative(time.time())}|{LoggerConvIndex.conversation_idx()}')
                    self.logger.debug(
                        "Interrupted, stopping text to speech after {} chunks".format(
                            chunk_idx
                        )
                    )
                    message_sent = (
                        f"{synthesis_result.get_message_up_to(seconds_spoken)}-"
                    )
                    cut_off = True
                    break
                if chunk_idx == 0:
                    if started_event:
                        started_event.set()

                self.output_device.consume_nonblocking(chunk_result.chunk)

                #TODO: The agent audio is sometimes clipping at the end. We have 
                # explored various ways to terminate the conversation while 
                # keeping in mind the recording but so far, the clipping remains and 
                # needs to be addressed.
                if self.recordings_dir is not None:                    
                    if self.first_time:
                        self.timestamp = datetime.utcnow().timestamp()
                    else:
                        current_timestamp = datetime.utcnow().timestamp()
                        diff_sec = int(current_timestamp - self.timestamp)
                        silent_chunk = self.synthesizer.create_silent_chunk(diff_sec)
                        self.timestamp = current_timestamp
                        self.synthesizer.send_audio(silent_chunk)
                        
                    self.synthesizer.send_audio(chunk_result.chunk)

                # @Roy: this is original vocode code and this is the rate-limiting.
                end_time = time.time()
                await asyncio.sleep(
                    max(
                        speech_length_seconds
                        - (end_time - start_time)
                        - self.per_chunk_allowance_seconds,
                        0,
                    )
                )

                self.logger.critical(
                    "Sent chunk {} with size {}".format(
                        chunk_idx, len(chunk_result.chunk)
                    )
                )
                if self.first_time:
                    self.first_time = False
                    self.first_synth_timestamp = datetime.utcnow().timestamp()
                self.mark_last_action_timestamp()
                chunk_idx += 1
                seconds_spoken += seconds_per_chunk
                if transcript_message:
                    transcript_message.text = synthesis_result.get_message_up_to(
                        seconds_spoken
                    )
            # we need to get the last chunk idx and timing it 
            if not is_filler_audio:
                sc_logger.info(f'tts_last_byte_time|{LoggerConvIndex.conversation_relative(time.time())}|{LoggerConvIndex.conversation_idx()}')
            if self.transcriber.get_transcriber_config().mute_during_speech:
                self.logger.debug("Unmuting transcriber")
                self.transcriber.unmute()
            if transcript_message:
                transcript_message.text = message_sent
            return message_sent, cut_off
        except Exception as exc:
            self.logger.exception(f"{exc}")
            self.is_audio_playing = False
        finally:
            self.is_audio_playing = False

    def mark_terminated(self):
        self.active = False

    def _log_call_transcript(self, transcript_as_string):
        # Log full call transcript in dynamo as a String attribute.
        conversation_id = self.id
        update_expression = 'SET callTranscript = :callTranscript'
        expression_attribute_values = {':callTranscript': transcript_as_string}
        dynamodb_table = DynamoDBTable('outboundCampaign-contactDetails',
                                       aws_profile_name=self.AWS_PROFILE_NAME)
        dynamodb_table.update_item(
            {'contactId': conversation_id},
            update_expression,
            expression_attribute_values
        )

    def merge_wav_files(self):
        """
        Utility function to align and merge the agent and human audio files.
        """
        # from synthesizer
        agent_file = f"{self.recordings_dir}/{self.id}_{AGENT_RECORDING_SUFFIX}.wav"

        # from telephony
        human_file = f"{self.recordings_dir}/{self.id}_{HUMAN_RECORDING_SUFFIX}.wav"
        output_file = f"{self.recordings_dir}/{self.id}_{ALL_RECORDING_SUFFIX}.wav"
        
        agent_sound = AudioSegment.from_wav(agent_file)
        human_sound = AudioSegment.from_wav(human_file)

        # AudioSegment.silent takes input time argument in milliseconds
        time_diff_ms = int((self.first_telephony_timestamp - self.first_synth_timestamp) * 1000)
        self.logger.debug(f"{self.first_telephony_timestamp}, {self.first_synth_timestamp} {time_diff_ms}")
        if time_diff_ms > 0:
            # synth/agent starts first, pad human/telephony file
            silence = AudioSegment.silent(duration=abs(time_diff_ms))
            human_sound = silence + human_sound
        elif time_diff_ms < 0:
            # telephony/human starts first, pad synth/agent file
            silence = AudioSegment.silent(duration=abs(time_diff_ms))
            agent_sound = silence + agent_sound
            
        # Find the length of the longest audio file
        longest_len = max(len(human_sound), len(agent_sound))

        # Pad the shorter audio with silence if necessary
        if len(human_sound) < longest_len:
            human_sound += AudioSegment.silent(duration=(longest_len - len(human_sound)))
        elif len(agent_sound) < longest_len:
            agent_sound += AudioSegment.silent(duration=(longest_len - len(agent_sound)))

        # Combine the two audio segments
        combined = AudioSegment.from_mono_audiosegments(human_sound, agent_sound)
        combined.export(output_file, format="wav")

    async def terminate(self):
            
        self.mark_terminated()
        self.broadcast_interrupt()
        self.events_manager.publish_event(
            TranscriptCompleteEvent(conversation_id=self.id, transcript=self.transcript)
        )
        
        if self.initial_step_task:
            self.logger.debug("Terminating initial_step Task")
            self.initial_step_task.cancel()
        if self.check_for_idle_task:
            self.logger.debug("Terminating check_for_idle Task")
            self.check_for_idle_task.cancel()
        if self.track_bot_sentiment_task:
            self.logger.debug("Terminating track_bot_sentiment Task")
            self.track_bot_sentiment_task.cancel()
        if self.events_manager and self.events_task:
            self.logger.debug("Terminating events Task")
            await self.events_manager.flush()
        self.logger.debug("Tearing down synthesizer")
        await self.synthesizer.tear_down()
        
        self.logger.debug("Terminating agent")
        if (
            isinstance(self.agent, ChatGPTAgent)
            and self.agent.agent_config.vector_db_config
        ):
            # Shutting down the vector db should be done in the agent's terminate method,
            # but it is done here because `vector_db.tear_down()` is async and
            # `agent.terminate()` is not async.
            self.logger.debug("Terminating vector db")
            await self.agent.vector_db.tear_down()
        
        transcript_as_string = self.agent.transcript.to_string()
        self.logger.info(f"TRANSCRIPT for {self.id}: {transcript_as_string}")
        self._log_call_transcript(transcript_as_string)
        self.agent.terminate()
        self.logger.debug("Terminating output device")
        self.output_device.terminate()
        self.logger.debug("Terminating speech transcriber")
        self.transcriber.terminate()
        self.logger.debug("Terminating transcriptions worker")
        self.transcriptions_worker.terminate()
        self.logger.debug("Terminating final transcriptions worker")
        self.agent_responses_worker.terminate()
        self.logger.debug("Terminating synthesis results worker")
        self.synthesis_results_worker.terminate()
        if self.filler_audio_worker is not None:
            self.logger.debug("Terminating filler audio worker")
            self.filler_audio_worker.terminate()
        if self.actions_worker is not None:
            self.logger.debug("Terminating actions worker")
            self.actions_worker.terminate()
        if self.recordings_dir is not None:                    
            self.audio_recorder.stop()
        self.logger.debug("Stopping audio recorder")
        #if self.recordings_dir is not None:
        #    self.merge_wav_files()

        self.logger.debug("Successfully terminated")

    def is_active(self):
        return self.active



def instantiate_logger(logger_name: str, 
                       format: str, 
                       level, 
                       in_memory:bool=True, 
                       file:str=None):
    
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    #memory_handler = logging.handlers.MemoryHandler(capacity=100, flushLevel=logging.ERROR, target=console_handler)
    #memory_handler.setFormatter(formatter)
    #logger.addHandler(memory_handler)

    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger