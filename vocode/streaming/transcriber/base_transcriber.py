from __future__ import annotations

import asyncio
import audioop
from opentelemetry import trace, metrics
from typing import Generic, TypeVar, Union, Optional
import secrets

from vocode import pubsub
from vocode.streaming.models.audio_encoding import AudioEncoding
from vocode.streaming.models.model import BaseModel
from vocode.streaming.constants import HUMAN_RECORDING_SUFFIX

from vocode.streaming.models.transcriber import TranscriberConfig
from vocode.streaming.utils.worker import AsyncWorker, ThreadAsyncWorker
from vocode.streaming.pubsub.base_pubsub import Publisher, PubSubTopics

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)


class Transcription(BaseModel):
    message: str
    confidence: float
    is_final: bool
    is_interrupt: bool = False
    def __str__(self):
        return f"Transcription({self.message}, {self.confidence}, {self.is_final})"


TranscriberConfigType = TypeVar("TranscriberConfigType", bound=TranscriberConfig)


class AbstractTranscriber(Generic[TranscriberConfigType]):
    def __init__(self, transcriber_config: TranscriberConfigType):
        self.transcriber_config = transcriber_config
        self.is_muted = False

    def mute(self):
        self.is_muted = True

    def unmute(self):
        self.is_muted = False

    def get_transcriber_config(self) -> TranscriberConfigType:
        return self.transcriber_config

    async def ready(self):
        return True

    def create_silent_chunk(self, chunk_size, sample_width=2):
        linear_audio = b"\0" * chunk_size
        if self.get_transcriber_config().audio_encoding == AudioEncoding.LINEAR16:
            return linear_audio
        elif self.get_transcriber_config().audio_encoding == AudioEncoding.MULAW:
            return audioop.lin2ulaw(linear_audio, sample_width)


class BaseAsyncTranscriber(AbstractTranscriber[TranscriberConfigType], AsyncWorker):
    def __init__(self, transcriber_config: TranscriberConfigType,
                 audio_id: Optional[str] = None):
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Transcription] = asyncio.Queue()
        self.publisher: Publisher = Publisher("BaseAsyncTranscriberPublisher")
        self.transcription_audio_id = audio_id or (
            f"transcription_audio_id_{secrets.token_urlsafe(16)}"
        )
        self.in_progress = False
        AsyncWorker.__init__(self, self.input_queue, self.output_queue)
        AbstractTranscriber.__init__(self, transcriber_config)

    async def _run_loop(self):
        raise NotImplementedError

    def is_in_progress(self):
        return False
    
    def send_audio(self, chunk):
        if self.publisher and self.transcriber_config.publish_audio:
            event_id = f"{self.transcription_audio_id}_{HUMAN_RECORDING_SUFFIX}"
            topic = PubSubTopics.INPUT_AUDIO_STREAMS
            _ = asyncio.create_task(
                self.publisher.publish(
                    event_id,
                    chunk,
                    self.transcriber_config.audio_encoding,
                    topic,
                    pubsub,
                )
            )
        self.consume_nonblocking(chunk)
        # TODO: create_silent_chunk(len(chunk)) rubbishes the input stream.  probably don't need.
        #
        # if not self.is_muted:
        #     self.consume_nonblocking(chunk)
        # else:
        #     self.consume_nonblocking(self.create_silent_chunk(len(chunk)))

    def terminate(self):
        termination_status = AsyncWorker.terminate(self)
        return termination_status


class BaseThreadAsyncTranscriber(
    AbstractTranscriber[TranscriberConfigType], ThreadAsyncWorker
):
    def __init__(self, transcriber_config: TranscriberConfigType):
        self.input_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Transcription] = asyncio.Queue()
        self.publisher: Publisher = Publisher("BaseThreadAsyncTranscriberPublisher")

        self.transcription_audio_id = (
            f"transcription_audio_id_{secrets.token_urlsafe(16)}"
        )
        ThreadAsyncWorker.__init__(self, self.input_queue, self.output_queue)
        AbstractTranscriber.__init__(self, transcriber_config)

    def _run_loop(self):
        raise NotImplementedError

    def send_audio(self, chunk):
        raise NotImplementedError


        if not self.is_muted:
            self.consume_nonblocking(chunk)
        else:
            self.consume_nonblocking(self.create_silent_chunk(len(chunk)))

    def terminate(self):
        ThreadAsyncWorker.terminate(self)


BaseTranscriber = Union[
    BaseAsyncTranscriber[TranscriberConfigType],
    BaseThreadAsyncTranscriber[TranscriberConfigType],
]
