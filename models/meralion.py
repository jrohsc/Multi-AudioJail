from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from .base import VoiceAssistant
from loguru import logger
import torch

class MERaLiONAssistant(VoiceAssistant):
    def __init__(self):
        repo_id = "MERaLiON/MERaLiON-AudioLLM-Whisper-SEA-LION"

        self.processor = AutoProcessor.from_pretrained(
            repo_id,
            trust_remote_code=True,
            cache_dir='/work/pi_ahoumansadr_umass_edu/jroh/jroh_umass_edu/.cache/huggingface/hub',
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            repo_id,
            use_safetensors=True,
            trust_remote_code=True,
            cache_dir='/work/pi_ahoumansadr_umass_edu/jroh/jroh_umass_edu/.cache/huggingface/hub',
            device_map='cuda',
            torch_dtype=torch.bfloat16,
        )

    def generate_audio(
        self,
        audio,
        max_new_tokens=2048,
    ):
        conversation = [
            {"role": "user", "content": "<SpeechHere>"}
        ]
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(text=chat_prompt, audios=audio['array']).to('cuda')
        inputs['input_features'] = inputs['input_features'].bfloat16()
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = outputs[:, inputs['input_ids'].size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_audio_defense(
        self,
        audio,
        DEFENSE_PROMPT,
        max_new_tokens=2048,
    ):
        conversation = [
            {"role": "system", "content": DEFENSE_PROMPT},
            {"role": "user", "content": "<SpeechHere>"}
        ]

        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(text=chat_prompt, audios=audio['array']).to('cuda')
        inputs['input_features'] = inputs['input_features'].bfloat16()
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = outputs[:, inputs['input_ids'].size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_audio_transcribe(
        self,
        audio,
        TRANSCRIPTION_PROMPT,
        max_new_tokens=2048,
    ):
        conversation = [
            {"role": "user", "content": "Please listen to the audio snippet carefully and transcribe the content."},
            {"role": "user", "content": "<SpeechHere>"}
        ]

        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(text=chat_prompt, audios=audio['array']).to('cuda')
        inputs['input_features'] = inputs['input_features'].bfloat16()
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = outputs[:, inputs['input_ids'].size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_text(self, text, max_new_tokens=2048):
        conversation = [
            {"role": "user", "content": text}  # Use the input text instead of "<SpeechHere>"
        ]
        chat_prompt = self.processor.tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process the text input (removed audio processing)
        inputs = self.processor(text=chat_prompt).to('cuda')
        
        # Ensure proper dtype conversion
        inputs['input_features'] = inputs['input_features'].bfloat16() if 'input_features' in inputs else None
        
        # Generate output
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Extract generated text
        generated_ids = outputs[:, inputs['input_ids'].size(1):]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
