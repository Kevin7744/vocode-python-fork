import logging

conversation_idx = 0
conversation_start_time = 0.0

class LoggerConvIndex(logging.Logger):

    num_human_events = 0
    num_agent_events = -1
    speech_final: bool = False
    def __init__(self, name, level=logging.NOTSET):
        super(LoggerConvIndex, self).__init__(name, level)

    @classmethod
    def idx_counter():
        global conversation_idx
        if LoggerConvIndex.speech_final and LoggerConvIndex.num_agent_events:
            conversation_idx += 1
            LoggerConvIndex.speech_final=False
            LoggerConvIndex.num_agent_events = 0

    @classmethod
    def conversation_idx(cls):
        global conversation_idx

        return conversation_idx
    
    @classmethod
    def next_turn(cls):
        global conversation_idx
        conversation_idx += 1

    @classmethod
    def set_conversation_start_time(cls, timestamp):
        global conversation_start_time
        conversation_start_time = timestamp

    @classmethod
    def get_conversation_start_time(cls, timestamp):
        global conversation_start_time
        return conversation_start_time

    @classmethod
    def conversation_relative(cls, timestamp):
        global conversation_start_time
        
        return timestamp - conversation_start_time

#logging.setLoggerClass(loggerConvIndex)
#logger = logging.getLogger(__name__)

#speech_final
#num_human_events
#num_agent_events
#if speech_final and num_agent_events > 0 --> conversation_idx + 1
