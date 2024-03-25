import asyncio
import json
import logging
from typing import Optional
import websockets
from websockets.client import WebSocketClientProtocol
import audioop
from urllib.parse import urlencode
from datetime import datetime 

from vocode import getenv
import time
from vocode.marrlabs.utils.logging_utils import LoggerConvIndex

from vocode.streaming.transcriber.base_transcriber import (
    BaseAsyncTranscriber,
    Transcription,
    meter,
)
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    EndpointingConfig,
    EndpointingType,
    PunctuationEndpointingConfig,
    TimeEndpointingConfig,
)
from vocode.streaming.models.audio_encoding import AudioEncoding


PUNCTUATION_TERMINATORS = [".", "!", "?", "..."]
NUM_RESTARTS = 5
RANDOM_CONF_THRESH = 0.7

avg_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.avg_latency",
    unit="seconds",
)
max_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.max_latency",
    unit="seconds",
)
min_latency_hist = meter.create_histogram(
    name="transcriber.deepgram.min_latency",
    unit="seconds",
)
duration_hist = meter.create_histogram(
    name="transcriber.deepgram.duration",
    unit="seconds",
)

logging.setLoggerClass(LoggerConvIndex)
logger_p = logging.getLogger(__name__+'_profiling')
# restart logging parameters
logging.setLoggerClass(logging.Logger)

class DeepgramTranscriber(BaseAsyncTranscriber[DeepgramTranscriberConfig]):
    def __init__(
        self,
        transcriber_config: DeepgramTranscriberConfig,
        api_key: Optional[str] = None,
        audio_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(transcriber_config, audio_id)
        self.api_key = api_key or getenv("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise Exception(
                "Please set DEEPGRAM_API_KEY environment variable or pass it as a parameter"
            )
        self._ended = False
        self.is_ready = False
        self.logger = logger or logging.getLogger(__name__)
        self.audio_cursor = 0.0
        self.in_progress = False
        self.transcription_started_event = asyncio.Event()

    async def _run_loop(self):
        restarts = 0
        while not self._ended and restarts < NUM_RESTARTS:
            await self.process()
            restarts += 1
            self.logger.debug(
                "Deepgram connection died, restarting, num_restarts: %s, ended: %s", restarts, self._ended
            )

    def send_audio(self, chunk):
        if (
            self.transcriber_config.downsampling
            and self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16
        ):
            chunk, _ = audioop.ratecv(
                chunk,
                2,
                1,
                self.transcriber_config.sampling_rate
                * self.transcriber_config.downsampling,
                self.transcriber_config.sampling_rate,
                None,
            )
        super().send_audio(chunk)

    def terminate(self):
        self.logger.debug(f"Termination attempt deepgram")
        # 
        terminate_msg = json.dumps({"type": "CloseStream"})
        self.input_queue.put_nowait(terminate_msg)
        self._ended = True
        termination_status = super().terminate()
        self.logger.debug(f"Termination status: {termination_status}")

    def get_deepgram_url(self):
        if self.transcriber_config.audio_encoding == AudioEncoding.LINEAR16:
            encoding = "linear16"
        elif self.transcriber_config.audio_encoding == AudioEncoding.MULAW:
            encoding = "mulaw"
        url_params = {
            "encoding": encoding,
            "sample_rate": self.transcriber_config.sampling_rate,
            "channels": 1,
            "interim_results": "true",
        }
        extra_params = {"endpointing": 10}
        if self.transcriber_config.vad_events:
            extra_params["vad_events"] = self.transcriber_config.vad_events
        if self.transcriber_config.language:
            extra_params["language"] = self.transcriber_config.language
        if self.transcriber_config.model:
            extra_params["model"] = self.transcriber_config.model
        if self.transcriber_config.tier:
            extra_params["tier"] = self.transcriber_config.tier
        if self.transcriber_config.version:
            extra_params["version"] = self.transcriber_config.version
        if self.transcriber_config.keywords:
            extra_params["keywords"] = self.transcriber_config.keywords
        if (
            self.transcriber_config.endpointing_config
            and self.transcriber_config.endpointing_config.type
            == EndpointingType.PUNCTUATION_BASED
        ):
            extra_params["punctuate"] = "true"
        
        url_params.update(extra_params)
        print(f"URL wss://api.deepgram.com/v1/listen?{urlencode(url_params)}")
        return f"wss://api.deepgram.com/v1/listen?{urlencode(url_params)}"

    def is_speech_final(
        self, current_buffer: str, deepgram_response: dict, time_silent: float
    ):
        transcript = deepgram_response["channel"]["alternatives"][0]["transcript"]

        # if it is not time based, then return true if speech is final and there is a transcript
        if not self.transcriber_config.endpointing_config:
            return transcript and deepgram_response["speech_final"]
        elif isinstance(
            self.transcriber_config.endpointing_config, TimeEndpointingConfig
        ):
            # if it is time based, then return true if there is no transcript
            # and there is some speech to send
            # and the time_silent is greater than the cutoff
            return (
                not transcript
                and current_buffer
                and (time_silent + deepgram_response["duration"])
                > self.transcriber_config.endpointing_config.time_cutoff_seconds
            )
        elif isinstance(
            self.transcriber_config.endpointing_config, PunctuationEndpointingConfig
        ):
            return (
                transcript
                and deepgram_response["speech_final"]
                and transcript.strip()[-1] in PUNCTUATION_TERMINATORS
            ) or (
                not transcript
                and current_buffer
                and (time_silent + deepgram_response["duration"])
                > self.transcriber_config.endpointing_config.time_cutoff_seconds
            )
        raise Exception("Endpointing config not supported")

    def calculate_time_silent(self, data: dict):
        end = data["start"] + data["duration"]
        words = data["channel"]["alternatives"][0]["words"]
        if words:
            return end - words[-1]["end"]
        return data["duration"]

    def is_in_progress(self):
        return self.in_progress

    async def process(self):
        self.audio_cursor = 0.0
        extra_headers = {"Authorization": f"Token {self.api_key}"}
        async with websockets.connect(
            self.get_deepgram_url(), extra_headers=extra_headers
        ) as ws:

            async def sender(ws: WebSocketClientProtocol):
                
                while not self._ended:
                    try:
                        # receive data from []
                        data = await asyncio.wait_for(self.input_queue.get(), 5)

                    except asyncio.exceptions.TimeoutError:
                        break

                    num_channels = 1
                    sample_width = 2
                    self.audio_cursor += len(data) / (
                        self.transcriber_config.sampling_rate
                        * num_channels
                        * sample_width
                    )
                    # send audio to deepgram
                    await ws.send(data)
                
                print('----------------------------------ENDED ALL')
                self.logger.debug(f"Terminating Deepgram transcriber sender, ended: {self._ended}")

            async def receiver(ws: WebSocketClientProtocol):
                buffer = ""
                # @Bilal I introduced this variable to get the interrupt to occur faster
                # Some improvement with it. 
                non_final_buffer = ""
                buffer_avg_confidence = 0
                num_buffer_utterances = 1
                time_silent = 0
                transcript_cursor = 0.0

                while not self._ended:
                    try:
                        msg = await ws.recv()
                    except Exception as e:
                        self.logger.debug(f"Got error {e} in Deepgram receiver")
                        break
                    data = json.loads(msg)
                    if data.get('type') == "SpeechStarted":
                        self.logger.critical(f"speech started {data}")
                        speech_start_time_vad = data['timestamp']
                        logger_p.info(f'speech_start_time_vad|{speech_start_time_vad}|{LoggerConvIndex.conversation_idx()}')
                        continue
                    elif (
                        not "is_final" in data
                    ):  # means we've finished receiving transcriptions
                        self.logger.debug("is_final not in deepgram data so breaking")
                        break
                    self.logger.debug(f"DEEPGRAM DATA {data}")
                    # total latency
                    # transcription latency = total - non-transcription
                    # (get the non-transcription latency by pinging deepgram)
                    cur_max_latency = self.audio_cursor - transcript_cursor
                    transcript_cursor = data["start"] + data["duration"]
                    cur_min_latency = self.audio_cursor - transcript_cursor

                    avg_latency_hist.record(
                        (cur_min_latency + cur_max_latency) / 2 * data["duration"]
                    )
                    duration_hist.record(data["duration"])

                    # Log max and min latencies
                    max_latency_hist.record(cur_max_latency)
                    min_latency_hist.record(max(cur_min_latency, 0))

                    is_final = data["is_final"]
                    speech_final = self.is_speech_final(buffer, data, time_silent)
                    top_choice = data["channel"]["alternatives"][0]
                    confidence = top_choice["confidence"]

                    if is_final:
                        if len(top_choice["words"]) > 0:
                            first_word = top_choice['words'][0]
                            first_word_txt = first_word['word']
                            last_word = top_choice['words'][-1]
                            last_word_txt = last_word['word']
                            speech_start_time = first_word['start']
                            logger_p.info(f'speech_start_time|{first_word_txt}|{speech_start_time}|{LoggerConvIndex.conversation_idx()}')
                        
                    if top_choice["transcript"] and confidence > 0.0:
                        self.logger.debug("top_choice['transcript'] and confidence > 0.0")
                        if not self.in_progress:
                            self.in_progress = True
                            self.transcription_started_event.set()

                        if is_final:
                            buffer = f"{buffer} {top_choice['transcript']}"
                            if buffer_avg_confidence == 0:
                                buffer_avg_confidence = confidence
                            else:
                                buffer_avg_confidence = (
                                    buffer_avg_confidence
                                    + confidence / (num_buffer_utterances)
                                ) * (num_buffer_utterances / (num_buffer_utterances + 1))
                            num_buffer_utterances += 1
                        elif confidence > RANDOM_CONF_THRESH:
                            non_final_buffer = top_choice['transcript']

                    if speech_final:
                        speech_end_time = last_word['end']
                        logger_p.info(f'asr_transcript(final): {buffer}')
                        logger_p.info(f'speech_end_time|{last_word_txt}|{speech_end_time}|{LoggerConvIndex.conversation_idx()}|{LoggerConvIndex.conversation_idx()}')
                        logger_p.info(f'asr_end_time|{LoggerConvIndex.conversation_relative(time.time())}|{LoggerConvIndex.conversation_idx()}')
                        self.output_queue.put_nowait(
                            Transcription(
                                message=buffer,
                                confidence=buffer_avg_confidence,
                                is_final=True,
                            )
                        )
                        buffer = ""
                        buffer_avg_confidence = 0
                        num_buffer_utterances = 1
                        time_silent = 0
                        self.in_progress = False
                    elif top_choice["transcript"] and confidence > 0.0:
                        logger_p.info(f'asr_transcript(high-confidence): {buffer}|{LoggerConvIndex.conversation_idx()}')
                        logger_p.info(f'asr_time_to_first_token|{LoggerConvIndex.conversation_relative(time.time())}|{LoggerConvIndex.conversation_idx()}')
                        self.output_queue.put_nowait(
                            Transcription(
                                message=buffer or non_final_buffer,
                                confidence=confidence,
                                is_final=False,
                            )
                        )
                        time_silent = self.calculate_time_silent(data)
                    else:
                        time_silent += data["duration"]

                self.logger.debug(f"Terminating Deepgram transcriber receiver, ended: {self._ended}")
            await asyncio.gather(sender(ws), receiver(ws))

