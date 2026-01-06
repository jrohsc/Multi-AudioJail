# Multilingual and Multi-Accent Jailbreaking of Audio LLMs

This is the official repository of "[Multilingual and Multi-Accent Jailbreaking of Audio LLMs]([https://arxiv.org/abs/2510.04201](https://arxiv.org/pdf/2504.01094?))". 

Our paper has been **published at COLM 2025 🎉**

![Multi-AudioJail Framework](fig/figure_1.png)

> **Multilingual and Multi-Accent Jailbreaking of Audio LLMs** <br>
> Jaechul Roh<sup>1</sup>, Virat Shejwalkar<sup>2</sup>, Amir Houmansadr<sup>2</sup>
> <br>
> <sup>1</sup>University of Massachusetts Amherst, <sup>2</sup>Google DeepMind <br>
>
> **Abstract.** Large Audio Language Models (LALMs) have significantly advanced audio
understanding but introduce critical security risks, particularly throughaudio jailbreaks. While prior work has focused on English-centric attacks,
we expose a far more severe vulnerability: adversarial multilingual and multiaccent audio jailbreaks, where linguistic and acoustic variations dramatically
amplify attack success. In this paper, we introduce MULTI-AUDIOJAIL, the
first systematic framework to exploit these vulnerabilities through (1) a
novel dataset of adversarially perturbed multilingual/multi-accent audio
jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing
that how acoustic perturbations (e.g., reverberation, echo, and whisper
effects) interacts with cross-lingual phonetics to cause jailbreak success
rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated
Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal
systems: attackers need only exploit the weakest link (e.g., non-English
audio inputs) to compromise the entire model, which we empirically show
by multilingual audio-only attacks achieving 3.1× higher success ratesthan text-only attacks. We plan to release our dataset to spur research into
cross-modal defenses, urging the community to address this expanding
attack surface in multimodality as LALMs evolve.
> 


## Key Findings

- **Audio vs. Text Vulnerability**: Audio-only attacks achieve 3.1× higher JSRs compared to text-only attacks
- **Multilingual Vulnerability**: German audio reaches 12.31% JSR vs. 3.92% for text
- **Perturbation Impact**: Reverberation causes JSRs to surge by up to +48.08 percentage points (e.g., Qwen2's German JSR: 9.71% → 57.79%)
- **Accent Vulnerability**: Kenyan accent (natural) increases by up to +57.25 points, reaching 61.25% JSR; Chinese accent (synthetic) increases by up to +55.87 points, reaching 59.75% JSR

## Models Evaluated

We evaluate five LALMs with low baseline JSRs from the [VoiceBench](https://github.com/MatthewCYM/VoiceBench) leaderboard:

| Model | Baseline JSR |
|-------|-------------|
| Qwen2-Audio | 3.27% |
| DiVA-llama-3-v0-8b | 1.73% |
| MERaLiON-AudioLLM-Whisper-SEA-LION | 5.19% |
| MiniCPM-o-2.6 | 2.31% |
| Ultravox-v0-4.1-Llama-3.1-8B | 3.08% |

## Dataset

Our comprehensive audio dataset comprises **102,720 audio files** based on 520 harmful instructions from AdvBench, organized into:

### Multilingual Category
- **Languages**: English (USA), German, Italian, Spanish, French, Portuguese
- Audio prompts generated directly in native languages via TTS

### Multi-Accent Category
- **Natural Accents**: Australia, Singapore, South Africa, Philippines, Kenya, Nigeria
- **Synthetic Accents**: China, Korea, Japan, Arabic, Portuguese, Spanish, Tamil

### Audio Perturbations
Five distinct perturbation techniques are applied:
1. **Reverb Teisco** - Resonant acoustic properties of teisco guitar performances
2. **Reverb Room** - Standard room acoustics (~0.6s reverberation time)
3. **Reverb Railway** - Complex reverberant conditions in railway environments
4. **Echo Effect** - Delayed repetition with attenuation factor
5. **Whisper Effect** - Amplitude reduction + high-frequency attenuation + breath noise

## Audio Perturbation Code

```python
# Reverb
def apply_reverb(input_audio, ir_audio, output_audio):
    x, sr = librosa.load(input_audio, sr=None)
    ir, _ = librosa.load(ir_audio, sr=sr)
    x_reverb = fftconvolve(x, ir, mode='full')
    x_reverb /= np.max(np.abs(x_reverb))
    sf.write(output_audio, x_reverb, sr)

# Echo
def add_echo(x, sr, delay=0.2, decay=0.5):
    delay_samples = int(sr * delay)
    echo_signal = np.zeros(len(x) + delay_samples)
    echo_signal[:len(x)] += x
    echo_signal[delay_samples:] += decay * x
    echo_signal /= np.max(np.abs(echo_signal))
    return echo_signal

# Whisper
def simulate_whisper(input_audio, output_audio, reduction_factor=0.3):
    x, sr = librosa.load(input_audio, sr=None)
    x_soft = x * reduction_factor
    x_whisper = high_freq_rolloff(x_soft, sr, cutoff=1500, order=4)
    x_whisper = add_breath_noise(x_whisper, sr, noise_level=0.005)
    x_whisper /= np.max(np.abs(x_whisper))
    sf.write(output_audio, x_whisper, sr)
```

## Results Summary

### Multilingual JSRs (%) with Reverb Teisco Perturbation

| Language | Qwen2 | DiVA | MERaLiON | MiniCPM | Ultravox | Avg. |
|----------|-------|------|----------|---------|----------|------|
| English | 22.88 (+20.96) | 14.62 (+13.66) | 17.98 (+13.08) | 17.98 (+16.73) | 14.62 (+13.56) | 17.62 (+15.60) |
| German | 57.79 (+48.08) | 34.71 (+24.71) | 44.71 (+24.04) | 22.88 (+7.30) | 47.79 (+42.21) | 41.58 (+29.27) |
| Italian | 50.19 (+41.25) | 34.71 (+30.77) | 31.25 (+21.15) | 47.12 (+40.68) | 39.33 (+36.06) | 40.52 (+33.98) |
| **Avg.** | **44.42 (+37.43)** | **27.68 (+23.48)** | **34.44 (+24.30)** | **27.42 (+20.64)** | **34.23 (+31.18)** | **33.64 (+27.41)** |

### Natural vs. Synthetic Accents
- **Natural accents**: Average JSR ~2.54% (baseline), up to 35.39% with perturbations
- **Synthetic accents**: Average JSR ~11.42% (baseline), up to 34.74% with perturbations

## Usage

See `demo.ipynb` for a complete demonstration of the attack and defense pipeline.

```python
from models import model_cls_mapping

# Load model
model = model_cls_mapping['meralion']()

# Generate response from audio
import librosa
audio_array, sr = librosa.load("audio.mp3", sr=16000)
audio_input = {"sampling_rate": sr, "array": audio_array}
response = model.generate_audio(audio_input)
```

## Project Structure

```
Multi-AudioJail/
├── demo.ipynb              # Demo notebook for attack and defense
├── models/                 # Model implementations
├── data/
│   ├── advbench_en/        # Standard adversarial audio samples
│   ├── advbench_en_reverb/ # Reverberated adversarial samples
│   └── advbench.csv        # Text prompts corresponding to audio
├── fig/
│   └── figure_1.png        # Framework overview figure
└── README.md
```

## Defense

We propose an inference-time, text-based defense method that leverages in-context learning by providing defense prompts during inference. Results show that applying defense generally reduces JSR:
- MERaLiON: -14.23 percentage points (German), -12.50 (Italian)
- Qwen2: -5.48% (German), -19.91% (Italian)

## Citation

```bibtex
@article{roh2025multilingual,
  title={Multilingual and multi-accent jailbreaking of audio llms},
  author={Roh, Jaechul and Shejwalkar, Virat and Houmansadr, Amir},
  journal={arXiv preprint arXiv:2504.01094},
  year={2025}
}
```

## Ethics Statement

This research is conducted exclusively to expose systemic risks in multimodal LLMs — not to facilitate misuse. We restrict the release of our full attack framework and instead publish only the curated adversarial dataset and audio modification methods. This enables the research community to develop defenses without providing malicious actors with turnkey exploit tools.

## Acknowledgments

The model implementations in `models/` are adapted from [VoiceBench](https://github.com/MatthewCYM/VoiceBench).

## License

This project is for research purposes only.
