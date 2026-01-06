from .diva import DiVAAssistant
from .qwen2 import Qwen2Assistant
from .ultravox import UltravoxAssistant
from .meralion import MERaLiONAssistant
from .minicpm import MiniCPMAssistant
from .ichigo import IchigoAssistant

model_cls_mapping = {
    'qwen2': Qwen2Assistant,
    'diva': DiVAAssistant,
    'ichigo': IchigoAssistant,
    'ultravox': UltravoxAssistant,
    'meralion': MERaLiONAssistant,
    'minicpm': MiniCPMAssistant,
}