import asyncio
import logging
import signal

from vocode.streaming.streaming_conversation import StreamingConversation
from vocode.helpers import create_streaming_microphone_input_and_speaker_output
from vocode.streaming.transcriber import *
from vocode.streaming.agent import *
from vocode.streaming.synthesizer import *
from vocode.streaming.models.transcriber import *
from vocode.streaming.models.agent import *
from vocode.streaming.models.synthesizer import *
from vocode.streaming.models.message import BaseMessage
import vocode
from TTS.api import TTS

# these can also be set as environment variables
vocode.setenv(
    OPENAI_API_KEY="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",
    DEEPGRAM_API_KEY="7e33fbf474d758c9cffede9a78da6805acea5013",
    AZURE_SPEECH_KEY="<your Azure key>",
    AZURE_SPEECH_REGION="<your Azure region>",
)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


async def main():
    (
        microphone_input,
        speaker_output,
    ) = create_streaming_microphone_input_and_speaker_output(
        use_default_devices=True,
        logger=logger,
    )

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
                initial_message=BaseMessage(text="What up"),
                prompt_preamble="""The AI is having a pleasant conversation about life""",
            )
        ),
            synthesizer=CoquiTTSSynthesizer(
                CoquiTTSSynthesizerConfig.from_output_device(
                speaker_output,
                tts_kwargs = {
                    "model_name": "tts_models/en/ljspeech/tacotron2-DDC_ph"
                }
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