from .base import VoiceAssistant
import transformers


class UltravoxAssistant(VoiceAssistant):
    def __init__(self):
        self.pipe = transformers.pipeline(model='fixie-ai/ultravox-v0_4_1-llama-3_1-8b', trust_remote_code=True, cache_dir='/work/pi_ahoumansadr_umass_edu/jroh/jroh_umass_edu/.cache/huggingface/hub', device='cuda')

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people."
            },
        ]
        return self.pipe({'audio': audio['array'], 'turns': turns, 'sampling_rate': audio['sampling_rate']}, max_new_tokens=max_new_tokens)

    def generate_audio_defense(
        self,
        audio,
        DEFENSE_PROMPT,
        max_new_tokens=2048,
    ):
        turns = [
            {
                "role": "system",
                "content": DEFENSE_PROMPT
            },
        ]
        return self.pipe({'audio': audio['array'], 'turns': turns, 'sampling_rate': audio['sampling_rate']}, max_new_tokens=max_new_tokens)

    def generate_audio_transcribe(
        self,
        audio,
        TRANSCRIPTION_PROMPT,
        max_new_tokens=2048,
    ):
        turns = [
            {
                "role": "user",
                "content": "You are a friendly and helpful character. You love to trascribe given audio. "
            },
        ]
        return self.pipe({'audio': audio['array'], 'turns': turns, 'sampling_rate': audio['sampling_rate']}, max_new_tokens=max_new_tokens)

    def generate_text(
        self,
        text,
        max_new_tokens=2048,
    ):
        turns = [
            {
                "role": "system",
                "content": "You are a friendly and helpful character. You love to answer questions for people."
            },
            {
                "role": "user",
                "content": text  # Use the input text instead of speech
            },
        ]
        
        return self.pipe({'turns': turns}, max_new_tokens=max_new_tokens)
