from .base import VoiceAssistant
from transformers import AutoModel, AutoTokenizer
import transformers
import torch
import soundfile as sf

class MiniCPMAssistant(VoiceAssistant):
    def __init__(self):
        self.model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-o-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa', # sdpa or flash_attention_2
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=False
        )
        self.model = self.model.eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-o-2_6', trust_remote_code=True)

        self.sys_prompt = self.model.get_sys_prompt(mode='audio_assistant', language='en')

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ): 
        user_question = {
            'role': 'user', 'content': [audio['array']]
        }
        msgs = [self.sys_prompt, user_question]
        
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=2048,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
        )

        return res

    def generate_audio_transcribe(
        self,
        audio,
        TRANSCRIPTION_PROMPT,
        max_new_tokens=2048,
    ): 
        user_question = {
            'role': 'user', 'content': "Please listen to the audio snippet carefully and transcribe the content.",
            'role': 'user', 'content': [audio['array']]
        }
        msgs = [self.sys_prompt, user_question]
        
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=2048,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
        )

        return res

    def generate_audio_defense(
        self,
        audio,
        DEFENSE_PROMPT,
        max_new_tokens=2048,
    ): 
        user_question = {
            'role': 'system', 'content': DEFENSE_PROMPT,
            'role': 'user', 'content': [audio['array']]
        }
        msgs = [self.sys_prompt, user_question]
        
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=2048,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
        )

        return res

    
    def generate_text(
        self,
        text_prompt,
        max_new_tokens=2048,
    ): 
        user_question = {
            'role': 'user', 'content': [text_prompt]
        }
        msgs = [self.sys_prompt, user_question]
        
        res = self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True,
            max_new_tokens=2048,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
        )

        return res