import random

import gradio as gr
import numpy as np
import torch

from src.chatterbox.enhanced_tts import (
    EnhancedTTS,
    GenerationResult,
    TTSGenerationConfig,
    create_enhanced_tts,
    generate_speech,
)
from src.chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS
from src.chatterbox.text_chunker import chunk_text_with_info, smart_chunk_text

# Device detection with MPS support for Apple Silicon Macs
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"🚀 Running on device: {DEVICE}")

# --- Global Model Initialization ---
MODEL = None
ENHANCED_TTS = None  # Enhanced TTS with text chunking support

LANGUAGE_CONFIG = {
    "ar": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ar_f/ar_prompts2.flac",
        "text": "في الشهر الماضي، وصلنا إلى معلم جديد بمليارين من المشاهدات على قناتنا على يوتيوب.",
    },
    "da": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/da_m1.flac",
        "text": "Sidste måned nåede vi en ny milepæl med to milliarder visninger på vores YouTube-kanal.",
    },
    "de": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/de_f1.flac",
        "text": "Letzten Monat haben wir einen neuen Meilenstein erreicht: zwei Milliarden Aufrufe auf unserem YouTube-Kanal.",
    },
    "el": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/el_m.flac",
        "text": "Τον περασμένο μήνα, φτάσαμε σε ένα νέο ορόσημο με δύο δισεκατομμύρια προβολές στο κανάλι μας στο YouTube.",
    },
    "en": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/en_f1.flac",
        "text": "Last month, we reached a new milestone with two billion views on our YouTube channel.",
    },
    "es": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac",
        "text": "El mes pasado alcanzamos un nuevo hito: dos mil millones de visualizaciones en nuestro canal de YouTube.",
    },
    "fi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fi_m.flac",
        "text": "Viime kuussa saavutimme uuden virstanpylvään kahden miljardin katselukerran kanssa YouTube-kanavallamme.",
    },
    "fr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/fr_f1.flac",
        "text": "Le mois dernier, nous avons atteint un nouveau jalon avec deux milliards de vues sur notre chaîne YouTube.",
    },
    "he": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/he_m1.flac",
        "text": "בחודש שעבר הגענו לאבן דרך חדשה עם שני מיליארד צפיות בערוץ היוטיוב שלנו.",
    },
    "hi": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/hi_f1.flac",
        "text": "पिछले महीने हमने एक नया मील का पत्थर छुआ: हमारे YouTube चैनल पर दो अरब व्यूज़।",
    },
    "it": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/it_m1.flac",
        "text": "Il mese scorso abbiamo raggiunto un nuovo traguardo: due miliardi di visualizzazioni sul nostro canale YouTube.",
    },
    "ja": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ja/ja_prompts1.flac",
        "text": "先月、私たちのYouTubeチャンネルで二十億回の再生回数という新たなマイルストーンに到達しました。",
    },
    "ko": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ko_f.flac",
        "text": "지난달 우리는 유튜브 채널에서 이십억 조회수라는 새로운 이정표에 도달했습니다.",
    },
    "ms": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ms_f.flac",
        "text": "Bulan lepas, kami mencapai pencapaian baru dengan dua bilion tontonan di saluran YouTube kami.",
    },
    "nl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/nl_m.flac",
        "text": "Vorige maand bereikten we een nieuwe mijlpaal met twee miljard weergaven op ons YouTube-kanaal.",
    },
    "no": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/no_f1.flac",
        "text": "Forrige måned nådde vi en ny milepæl med to milliarder visninger på YouTube-kanalen vår.",
    },
    "pl": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pl_m.flac",
        "text": "W zeszłym miesiącu osiągnęliśmy nowy kamień milowy z dwoma miliardami wyświetleń na naszym kanale YouTube.",
    },
    "pt": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/pt_m1.flac",
        "text": "No mês passado, alcançámos um novo marco: dois mil milhões de visualizações no nosso canal do YouTube.",
    },
    "ru": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/ru_m.flac",
        "text": "В прошлом месяце мы достигли нового рубежа: два миллиарда просмотров на нашем YouTube-канале.",
    },
    "sv": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sv_f.flac",
        "text": "Förra månaden nådde vi en ny milstolpe med två miljarder visningar på vår YouTube-kanal.",
    },
    "sw": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/sw_m.flac",
        "text": "Mwezi uliopita, tulifika hatua mpya ya maoni ya bilioni mbili kweny kituo chetu cha YouTube.",
    },
    "tr": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/tr_m.flac",
        "text": "Geçen ay YouTube kanalımızda iki milyar görüntüleme ile yeni bir dönüm noktasına ulaştık.",
    },
    "zh": {
        "audio": "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/zh_f2.flac",
        "text": "上个月，我们达到了一个新的里程碑。 我们的YouTube频道观看次数达到了二十亿次，这绝对令人难以置信。",
    },
}


# --- UI Helpers ---
def default_audio_for_ui(lang: str) -> str | None:
    return LANGUAGE_CONFIG.get(lang, {}).get("audio")


def default_text_for_ui(lang: str) -> str:
    return LANGUAGE_CONFIG.get(lang, {}).get("text", "")


def get_supported_languages_display() -> str:
    """Generate a formatted display of all supported languages."""
    language_items = []
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        language_items.append(f"**{name}** (`{code}`)")

    # Split into 2 lines
    mid = len(language_items) // 2
    line1 = " • ".join(language_items[:mid])
    line2 = " • ".join(language_items[mid:])

    return f"""
### 🌍 Supported Languages ({len(SUPPORTED_LANGUAGES)} total)
{line1}

{line2}
"""


def get_or_load_model():
    """Loads the ChatterboxMultilingualTTS model if it hasn't been loaded already,
    and ensures it's on the correct device."""
    global MODEL, ENHANCED_TTS
    if MODEL is None:
        print("Model not loaded, initializing...")
        try:
            # Convert string device to torch.device object
            device_obj = torch.device(DEVICE)
            MODEL = ChatterboxMultilingualTTS.from_pretrained(device_obj)
            if hasattr(MODEL, "to") and str(MODEL.device) != DEVICE:
                MODEL.to(device_obj)

            # Initialize enhanced TTS with chunking support
            if ENHANCED_TTS is None:
                ENHANCED_TTS = create_enhanced_tts(
                    model=MODEL,
                    device=DEVICE,
                    logger=None,  # Use default logger
                )
                print("Enhanced TTS with text chunking initialized")

            print(
                f"Model loaded successfully. Internal device: {getattr(MODEL, 'device', 'N/A')}"
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return MODEL


# Attempt to load the model at startup.
try:
    get_or_load_model()
except Exception as e:
    print(
        f"CRITICAL: Failed to load model on startup. Application may not function. Error: {e}"
    )


def set_seed(seed: int):
    """Sets the random seed for reproducibility across torch, numpy, and random."""
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    elif DEVICE == "mps":
        # MPS (Metal Performance Shaders) doesn't require specific seeding
        # but we keep the structure for consistency
        pass
    random.seed(seed)
    np.random.seed(seed)


def resolve_audio_prompt(language_id: str, provided_path: str | None) -> str | None:
    """
    Decide which audio prompt to use:
    - If user provided a path (upload/mic/url), use it.
    - Else, fall back to language-specific default (if any).
    """
    if provided_path and str(provided_path).strip():
        return provided_path
    return LANGUAGE_CONFIG.get(language_id, {}).get("audio")


def generate_tts_audio(
    text_input: str,
    language_id: str,
    audio_prompt_path_input: str = None,
    exaggeration_input: float = 0.5,
    temperature_input: float = 0.8,
    seed_num_input: int = 0,
    cfgw_input: float = 0.5,
) -> tuple[int, np.ndarray]:
    """
    Generate high-quality speech audio from text using enhanced Chatterbox Multilingual model with intelligent text chunking.
    Supports long texts by automatically splitting them at natural break points and concatenating the generated audio.

    This enhanced version uses smart text chunking to handle texts longer than 300 characters while maintaining
    natural speech flow. Each chunk is generated separately and then seamlessly concatenated together.

    Args:
        text_input (str): The text to synthesize into speech (supports unlimited length with automatic chunking)
        language_id (str): The language code for synthesis (supports all 23 languages)
        audio_prompt_path_input (str, optional): File path or URL to the reference audio file that defines the target voice style. Defaults to None.
        exaggeration_input (float, optional): Controls speech expressiveness (0.25-2.0, neutral=0.5, extreme values may be unstable). Defaults to 0.5.
        temperature_input (float, optional): Controls randomness in generation (0.05-5.0, higher=more varied). Defaults to 0.8.
        seed_num_input (int, optional): Random seed for reproducible results (0 for random generation). Defaults to 0.
        cfgw_input (float, optional): CFG/Pace weight controlling generation guidance (0.2-1.0). Defaults to 0.5, 0 for language transfer.

    Returns:
        tuple[int, np.ndarray]: A tuple containing the sample rate (int) and the generated audio waveform (numpy.ndarray)
    """
    global ENHANCED_TTS

    # Load model and initialize enhanced TTS if needed
    if ENHANCED_TTS is None:
        current_model = get_or_load_model()
        if current_model is None:
            raise RuntimeError("TTS model failed to load.")

    if ENHANCED_TTS is None:
        raise RuntimeError("Enhanced TTS system failed to initialize.")

    if seed_num_input != 0:
        set_seed(int(seed_num_input))

    print(
        f"🚀 Generating audio with enhanced TTS for text: '{text_input[:50]}{'...' if len(text_input) > 50 else ''}' (length: {len(text_input)} chars)"
    )

    # Create generation configuration for enhanced TTS
    config = TTSGenerationConfig(
        max_chars=300,  # Chunk size
        language_id=language_id,
        exaggeration=exaggeration_input,
        temperature=temperature_input,
        cfg_weight=cfgw_input,
        show_progress=True,
        enable_tqdm=True,
        concatenate_audio=True,
        add_silence_between_chunks=0.05,  # 50ms silence between chunks
    )

    try:
        # Use enhanced TTS for generation with automatic chunking
        result = ENHANCED_TTS.generate(text_input, config)

        print(
            f"✅ Audio generation complete! Generated {result.chunk_count} chunks, duration: {result.duration:.2f}s"
        )

        # Return in the expected format (sample_rate, audio_data)
        return (result.sample_rate, result.audio_data.squeeze(0))

    except Exception as e:
        print(f"❌ Enhanced TTS generation failed: {str(e)}")
        print("🔄 Falling back to basic model generation...")

        # Fallback to original model for compatibility
        current_model = get_or_load_model()
        if current_model is None:
            raise RuntimeError("TTS model is not loaded.")

        # Handle optional audio prompt for fallback
        chosen_prompt = audio_prompt_path_input or default_audio_for_ui(language_id)

        generate_kwargs = {
            "exaggeration": exaggeration_input,
            "temperature": temperature_input,
            "cfg_weight": cfgw_input,
        }
        if chosen_prompt:
            generate_kwargs["audio_prompt_path"] = chosen_prompt
            print(f"Using audio prompt: {chosen_prompt}")
        else:
            print("No audio prompt provided; using default voice.")

        # Truncate text to 300 chars for fallback
        fallback_text = text_input[:300]
        if len(text_input) > 300:
            print(f"⚠️ Text truncated to 300 characters for fallback generation")

        wav = current_model.generate(
            fallback_text,
            language_id=language_id,
            **generate_kwargs,
        )
        print("Fallback audio generation complete.")
        return (current_model.sr, wav.squeeze(0).numpy())


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Chatterbox Multilingual Demo
        Generate high-quality multilingual speech from text with reference audio styling, supporting 23 languages.

        For a hosted version of Chatterbox Multilingual and for finetuning, please visit [resemble.ai](https://app.resemble.ai)
        """
    )

    # Display supported languages
    gr.Markdown(get_supported_languages_display())
    with gr.Row():
        with gr.Column():
            initial_lang = "fr"
            text = gr.Textbox(
                value=default_text_for_ui(initial_lang),
                label="Text to synthesize (unlimited length with automatic chunking)",
                max_lines=5,
            )

            language_id = gr.Dropdown(
                choices=list(
                    ChatterboxMultilingualTTS.get_supported_languages().keys()
                ),
                value=initial_lang,
                label="Language",
                info="Select the language for text-to-speech synthesis",
            )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference Audio File (Optional)",
                value=default_audio_for_ui(initial_lang),
            )

            gr.Markdown(
                "💡 **Note**: Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip's language. To mitigate this, set the CFG weight to 0.",
                elem_classes=["audio-note"],
            )

            exaggeration = gr.Slider(
                0.25,
                2,
                step=0.05,
                label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
                value=0.5,
            )
            cfg_weight = gr.Slider(0.2, 1, step=0.05, label="CFG/Pace", value=0.5)

            with gr.Accordion("More options", open=False):
                seed_num = gr.Number(value=0, label="Random seed (0 for random)")
                temp = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Output Audio")

        def on_language_change(lang, current_ref, current_text):
            return default_audio_for_ui(lang), default_text_for_ui(lang)

        language_id.change(
            fn=on_language_change,
            inputs=[language_id, ref_wav, text],
            outputs=[ref_wav, text],
            show_progress=False,
        )

    run_btn.click(
        fn=generate_tts_audio,
        inputs=[
            text,
            language_id,
            ref_wav,
            exaggeration,
            temp,
            seed_num,
            cfg_weight,
        ],
        outputs=[audio_output],
    )

demo.launch()
