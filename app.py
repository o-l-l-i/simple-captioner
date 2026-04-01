import os
import torch
import gradio as gr
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import time


IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
VIDEO_EXTENSIONS = ('.mp4', '.mov', '.avi', '.webm', '.mkv', ".gif", ".flv")
DEFAULT_PROMPT = "Describe this media."
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_QUANT = "8-bit"  # "None" | "8-bit" | "4-bit"
DEFAULT_ATTN = "eager" if os.name == "nt" else "flash_attention_2"


PRESETS = {
    "auto": {
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 896 * 28 * 28,
    },
    "fast": {
        "resized_height": 392,
        "resized_width": 392,
    },
    "high": {
        "resized_height": 728,
        "resized_width": 728,
    },
}


AVAILABLE_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "Custom...",
]


# Globals
processor = None
current_model_id = DEFAULT_MODEL_ID
current_quant = DEFAULT_QUANT
model = None
should_abort = False
ui_e = {}


def build_bnb_config(quant_choice: str):
    if quant_choice == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant_choice == "4-bit":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    return None

def is_qwen35_model(model_id: str) -> bool:
    """Qwen3.5 models use a different class and require thinking-mode handling."""
    return "Qwen3.5" in model_id or "Qwen3_5" in model_id


def pick_model_class(model_id: str):
    from transformers import AutoModelForImageTextToText
    try:
        if is_qwen35_model(model_id):
            from transformers import Qwen3_5ForConditionalGeneration
            return Qwen3_5ForConditionalGeneration
        if "Qwen3-VL" in model_id:
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration
        if "Qwen2.5-VL" in model_id or "Qwen2_5-VL" in model_id:
            from transformers import Qwen2_5_VLForConditionalGeneration
            return Qwen2_5_VLForConditionalGeneration
    except Exception:
        pass
    return AutoModelForImageTextToText

def unload_model():

    print ("[DEBUG] Unload currently loaded model")
    global model, processor
    try:
        del model
    except Exception:
        pass
    try:
        del processor
    except Exception:
        pass
    model = None
    processor = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_selected_model(model_id: str, quant_choice: str, attn_impl: str = DEFAULT_ATTN):
    """
    Loads (or reloads) the model + processor with chosen quantization and attention impl.
    Falls back to 'eager' if flash_attention_2 fails.
    """

    print("[DEBUG] Loading selected model:", model_id)
    global model, processor, current_model_id, current_quant

    unload_model()

    kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
        "attn_implementation": attn_impl,
    }
    bnb = build_bnb_config(quant_choice)
    if bnb is not None:
        kwargs["quantization_config"] = bnb

    ModelCls = pick_model_class(model_id)
    try:
        model = ModelCls.from_pretrained(model_id, **kwargs)
    except Exception:
        if attn_impl == "flash_attention_2":
            kwargs["attn_implementation"] = "eager"
            model = ModelCls.from_pretrained(model_id, **kwargs)
        else:
            raise
    from transformers import AutoProcessor as _AP
    processor = _AP.from_pretrained(model_id)

    current_model_id = model_id
    current_quant = quant_choice
    return get_model_info()


def toggle_controls(disabled=True):
    updates = {}
    for name, component in ui_e.items():
        if name == "abort_button":
            updates[name] = gr.update(interactive=not disabled)
        else:
            updates[name] = gr.update(interactive=not disabled if component.visible else False)
    return updates


def disable_controls_dict():
    return [toggle_controls(disabled=True)[k] for k in control_keys]

def enable_controls_dict():
    return [toggle_controls(disabled=False)[k] for k in control_keys]

control_keys = [
    "model_dropdown",
    "quant_dropdown",
    "attn_dropdown",
    "load_button",
    "reset_button",
    "start_button",
    "abort_button",
    "folder_input",
    "prompt_input",
    "skip_existing_checkbox",
    "max_tokens_slider",
    "summary_mode",
    "one_sentence_mode",
    "retain_preview_checkbox",
    "resolution_mode",
    "status_output",
]


def finish_process():
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gr.update(interactive=False)
    return updates


def abort_process():
    global should_abort
    should_abort = True
    updates = enable_controls_dict()
    abort_index = control_keys.index("abort_button")
    status_index = control_keys.index("status_output")
    updates[abort_index] = gr.update(interactive=False)
    updates[status_index] = gr.update(value="⛔ Aborting process...")
    return updates


def start_process():
    updates = disable_controls_dict()
    abort_index = control_keys.index("abort_button")
    updates[abort_index] = gr.update(interactive=True)
    return updates


def serialize_for_debug(obj):
    if isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(i) for i in obj]
    elif isinstance(obj, Image.Image):
        return f"<Image {obj.size} {obj.mode}>"
    else:
        return obj


def get_model_info():
    global model
    if model is None:
        return "Model not loaded.", "N/A", "N/A", "N/A", "N/A"

    model_name = model.config._name_or_path if hasattr(model.config, "_name_or_path") else "Unknown Model"
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    if torch.cuda.is_available():
        vram_used = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        vram_total = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    else:
        vram_used, vram_total = "N/A", "N/A"

    dtype = str(next(model.parameters()).dtype)
    return model_name, device, f"{vram_used} / {vram_total}", dtype, str(model.config)


print("[DEBUG] Cuda available:", torch.cuda.is_available())

def generate_caption(media_path, prompt, max_tokens, summary_mode=False, one_sentence_mode=False, resolution_mode="auto"):
    global processor, model
    assert model is not None, "Model must be loaded before generating captions."
    assert processor is not None, "Processor must be loaded before generating captions."
    # Edit custom prompts here if you want
    if summary_mode and one_sentence_mode:
        prompt += " Give a one-sentence summary of the scene."
    elif summary_mode:
        prompt += " Give a short summary of the scene."
    elif one_sentence_mode:
        prompt += " Describe this image in one sentence."

    ext = os.path.splitext(media_path)[-1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    content_type = "video" if is_video else "image"


    if is_video:
        media_data = media_path
    else:
        media_data = Image.open(media_path).convert("RGB")


    content_block = {"type": content_type, content_type: media_data}

    if not is_video:

        if resolution_mode == "auto":
            content_block["min_pixels"] = 256 * 28 * 28
            content_block["max_pixels"] = 896 * 28 * 28
        elif resolution_mode == "auto_high":
            content_block["min_pixels"] = 256 * 28 * 28
            content_block["max_pixels"] = 1280 * 28 * 28
        elif resolution_mode == "fast":
            content_block["resized_height"] = 392
            content_block["resized_width"] = 392
        elif resolution_mode == "high":
            content_block["resized_height"] = 728
            content_block["resized_width"] = 728


    messages = [
        {
            "role": "user",
            "content": [
                content_block,
                {"type": "text", "text": prompt},
            ],
        }
    ]

    qwen35 = is_qwen35_model(current_model_id)
    if qwen35:
        messages.append({"role": "assistant", "content": "<think>\n\n</think>\n\n"})

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not qwen35,
        continue_final_message=qwen35,
    )

    vision_info = process_vision_info(messages)

    assert isinstance(vision_info, tuple), "process_vision_info did not return a tuple"

    if len(vision_info) == 3:
        image_inputs, video_inputs, _ = vision_info
    elif len(vision_info) == 2:
        image_inputs, video_inputs = vision_info # type: ignore
    else:
        raise ValueError(f"Error: Expected 2 or 3 values from process_vision_info, but got {len(vision_info)}.")

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")


    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    caption = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    if qwen35 and "<think>" in caption:
        import re
        caption = re.sub(r"<think>.*?</think>\s*", "", caption, flags=re.DOTALL)

    return caption.strip()


def is_image_file(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def is_video_file(filename):
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def build_final_prompt(user_prompt, summary, one_sentence):
    parts = [user_prompt.strip()]
    if summary:
        parts.append("Please provide a short summary.")
    if one_sentence:
        parts.append("Keep the description to one sentence.")
    return " ".join(parts)


def process_folder(folder_path, prompt, skip_existing, max_tokens, summary_mode, one_sentence_mode, retain_preview, resolution_mode):

    global should_abort, current_model_id, current_quant

    processed_media = 0
    skipped_media = 0
    failed_media = 0
    last_media_to_show = None
    last_caption = ""
    last_media_name_markdown = ""
    elapsed_str = ""

    print("[DEBUG] starting folder processing...:")
    print("[DEBUG] using model: ", current_model_id)
    print("[DEBUG] model quantization: ", current_quant)
    print("[DEBUG] folder_path: ", folder_path)
    print("[DEBUG] prompt: ", prompt)
    print("[DEBUG] skip_existing: ", skip_existing)
    print("[DEBUG] max_tokens: ", max_tokens)
    print("[DEBUG] summary_mode: ", summary_mode)
    print("[DEBUG] one_sentence_mode: ", one_sentence_mode)
    print("[DEBUG] retain_preview: ", retain_preview)
    print("[DEBUG] resolution_mode: ", resolution_mode)
    print("[DEBUG] should_abort: ", should_abort)

    if not folder_path.strip():
        control_updates = finish_process()
        yield (
            "⚠️ Please enter a valid folder path.",
            None,
            None,
            "No media to process.",
            0,
            "",
            *control_updates
        )
        return

    if not os.path.exists(folder_path):
        control_updates = finish_process()
        yield (
            f"❌ Folder not found: {folder_path}",
            None,
            None,
            "No media to process.",
            0,
            "",
            *control_updates
        )
        return

    media_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:

            if is_image_file(file) or is_video_file(file):
                media_files.append(os.path.join(root, file))

    total_media = len(media_files)

    if total_media == 0:
        control_updates = finish_process()
        yield (
            "📂 No media found in the folder or subfolders.",
            None,
            None,
            "No media to process.",
            0,
            "",
            *control_updates
        )
        return

    start_time = time.time()

    for idx, media_path in enumerate(media_files):
        if should_abort:
            should_abort = False
            status_index = control_keys.index("status_output")
            abort_index = control_keys.index("abort_button")
            control_updates = enable_controls_dict()
            control_updates[status_index] = gr.update(value="⛔ Aborted by user.")
            control_updates[abort_index] = gr.update(interactive=False)

            yield (
                "⛔ Aborted by user.",
                None,
                None,
                "Aborted.",
                0,
                "",
                *control_updates
            )
            return
        try:
            txt_filename = os.path.splitext(os.path.basename(media_path))[0] + ".txt"
            txt_path = os.path.join(os.path.dirname(media_path), txt_filename)

            rel_path = os.path.relpath(media_path, folder_path)
            media_name_markdown = f"**File:** `{rel_path}`"

            if skip_existing and os.path.exists(txt_path):
                progress = int(((idx + 1) / total_media) * 100)
                elapsed = int(time.time() - start_time)
                elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"
                skipped_media += 1

                control_updates = start_process()
                yield (
                    f"⏭️ Skipped {idx+1}/{total_media}: {rel_path} (already captioned)",
                    last_media_to_show if retain_preview else None,
                    last_media_name_markdown if retain_preview else None,
                    last_caption if retain_preview else "Skipped (already captioned)",
                    progress,
                    elapsed_str,
                    *control_updates
                )
                continue

            caption = generate_caption(media_path, prompt, max_tokens, summary_mode, one_sentence_mode, resolution_mode)

            # Only generate previews for images
            if is_image_file(media_path):
                media_to_show = Image.open(media_path)
            else:
                media_to_show = None

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)

            progress = int(((idx + 1) / total_media) * 100)
            elapsed = int(time.time() - start_time)
            elapsed_str = f"{elapsed//60:02d}:{elapsed%60:02d}"

            last_media_to_show = media_to_show
            last_caption = caption
            last_media_name_markdown = media_name_markdown
            processed_media += 1
            control_updates = start_process()
            yield (
                f"🖼️ Processing {idx+1}/{total_media}: {rel_path}",
                media_to_show,
                media_name_markdown,
                caption,
                progress,
                elapsed_str,
                *control_updates
            )

        except Exception as e:
            failed_media += 1
            print(f"[ERROR] Failed processing file {media_path}: {e}")
            control_updates = start_process()
            yield (
                f"⚠️ Error processing {media_path}: {str(e)}",
                None,
                None,
                "Error in captioning.",
                0,
                elapsed_str,
                *control_updates
            )

    end_time = time.time()
    total_time = f"{end_time - start_time:.2f}"


    control_updates = finish_process()
    yield (
        f"✅ Processing complete! , processed {processed_media} media in {elapsed_str}, skipped {skipped_media} media. Failed to process {failed_media} media (inaccessible, unknown or broken file)",
        last_media_to_show,
        last_media_name_markdown,
        last_caption,
        None,
        None,
        *control_updates
    )


def reset_prompt():
    return DEFAULT_PROMPT


css = """
.generating {
    border: none;
}
"""


with gr.Blocks() as iface: # type: ignore

    gr.Markdown("# Simple Captioner")
    gr.Markdown("A simple media caption generator for images and video using **[Qwen2.5/3/3.5 VL Instruct](https://huggingface.co/Qwen/)**")
    gr.Markdown("Supported image formats: png, jpg, jpeg, bmp, gif, webp")
    gr.Markdown("Supported video formats: mp4, mov, avi, webm, mkv, gif, flv")
    gr.Markdown("Written by [Olli S.](https://github.com/o-l-l-i)")


    with gr.Accordion("⚙️ Model Settings", open=True):
        ui_e["model_dropdown"] = model_dropdown = gr.Dropdown(
            label="Model",
            choices=AVAILABLE_MODELS,
            value=DEFAULT_MODEL_ID,
            allow_custom_value=False,
            interactive=True,
            info="Pick a model to use for captioning."
        )
        custom_model_box = gr.Textbox(
            label="Custom Model ID (Hugging Face)",
            placeholder="e.g. Qwen/Qwen3-VL-4B-Instruct or your-org/my-qwen3-checkpoint",
            visible=False,
        )
        ui_e["quant_dropdown"] = quant_dropdown = gr.Radio(
            label="Quantization",
            choices=["None", "8-bit", "4-bit"],
            value=DEFAULT_QUANT,
            interactive=True,
            info="Lower-bit quantization reduces VRAM, may slightly affect quality."
        )
        ui_e["attn_dropdown"] = attn_dropdown = gr.Radio(
            label="Attention Implementation",
            choices=["flash_attention_2", "eager"],
            value=DEFAULT_ATTN,
            interactive=True,
            info="If FlashAttention isn't installed/working, choose 'eager'. Auto-fallback on load."
        )
        ui_e["load_button"] = gr.Button("📦 Load / Reload Model")

    def _toggle_custom(choice):
        return gr.update(visible=(choice == "Custom…"))

    model_dropdown.change(_toggle_custom, inputs=[model_dropdown], outputs=[custom_model_box])

    def _ui_load_model(sel, custom_id, quant, attn):
        model_id = custom_id.strip() if sel == "Custom..." and custom_id and custom_id.strip() else sel
        name, device, vram, dtype, cfg = load_selected_model(model_id, quant, attn)
        status = f"✅ Loaded '{model_id}' with {quant} quantization ({attn})."
        print ("[DEBUG]", status)
        return status, name, device, vram, dtype, cfg
    with gr.Accordion("⚙️ Model Information", open=False):
        model_name_display = gr.Textbox(label="Model Name", interactive=False)
        device_display = gr.Textbox(label="Device", interactive=False)
        vram_display = gr.Textbox(label="VRAM Usage", interactive=False)
        dtype_display = gr.Textbox(label="Torch Dtype", interactive=False)
        config_display = gr.Textbox(label="Model Config", interactive=False, lines=4)

    with gr.Row():
        ui_e["folder_input"] = gr.Textbox(label="📁 Folder Path", placeholder="e.g. C:\\Users\\you\\Pictures\\input_images")
        ui_e["prompt_input"] = gr.Textbox(label="Custom Prompt", value=DEFAULT_PROMPT)

    ui_e["skip_existing_checkbox"] = gr.Checkbox(label="Skip already captioned media (.txt exists)", value=False)

    with gr.Row():
        gr.Markdown("### Prompt Controls")
        gr.Markdown("""
        - **Summary Mode**: Asks the model to summarize the media content briefly.
        - **One-Sentence Mode**: Instructs the model to keep the caption to a single concise sentence.
        """)
        ui_e["summary_mode"] = gr.Checkbox(label="Summary Mode", value=False)
        ui_e["one_sentence_mode"] = gr.Checkbox(label="One-Sentence Mode", value=False)

    prompt_preview = gr.Textbox(
        label="Final Prompt Preview",
        lines=2,
        interactive=False
    )

    with gr.Row():
        ui_e["max_tokens_slider"] = gr.Slider(label="🧾 Max Tokens", minimum=32, maximum=512, value=128, step=16)
        ui_e["resolution_mode"] = gr.Dropdown(
            label="Image Resolution",
            choices=["auto", "auto_high", "fast", "high"],
            value="auto",
            info="Choose the resolution mode for visual input."
        )

    with gr.Row():
        ui_e["reset_button"] = gr.Button("🔄 Reset to Default Prompt")
        ui_e["start_button"] = gr.Button("🚀 Start Processing", interactive=True)
        ui_e["abort_button"] = gr.Button("⛔ Abort", interactive=False)

    ui_e["status_output"] = gr.Textbox(label="Status", interactive=False)
    progress_bar = gr.Slider(minimum=0, maximum=100, label="Progress", interactive=False)
    time_display = gr.Textbox(label="⏱️ Time Taken (s)", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            media_output = gr.Image(label="Current Image", interactive=False)
            media_name_markdown = gr.Markdown()
            ui_e["retain_preview_checkbox"] = gr.Checkbox(
                label="Retain preview on skip",
                value=True
            )
        with gr.Column(scale=1):
            caption_output = gr.Textbox(label="Generated Caption", interactive=False)

    ui_e["prompt_input"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["summary_mode"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["one_sentence_mode"].change(
        fn=build_final_prompt,
        inputs=[ui_e["prompt_input"], ui_e["summary_mode"], ui_e["one_sentence_mode"]],
        outputs=[prompt_preview],
    )

    ui_e["start_button"].click(
        start_process,
        inputs=[],
        outputs=[ui_e[k] for k in control_keys]
    )

    ui_e["start_button"].click(
        process_folder,
        inputs=[
            ui_e["folder_input"],
            ui_e["prompt_input"],
            ui_e["skip_existing_checkbox"],
            ui_e["max_tokens_slider"],
            ui_e["summary_mode"],
            ui_e["one_sentence_mode"],
            ui_e["retain_preview_checkbox"],
            ui_e["resolution_mode"],
        ],
        outputs=[
            ui_e["status_output"],
            media_output,
            media_name_markdown,
            caption_output,
            progress_bar,
            time_display,
            *[ui_e[k] for k in control_keys]
        ]
    )

    ui_e["abort_button"].click(
        fn=abort_process,
        inputs=[],
        outputs=[ui_e[k] for k in control_keys],
        queue=False
    )

    ui_e["reset_button"].click(reset_prompt, inputs=[], outputs=[ui_e["prompt_input"]])
    ui_e["start_button"].click(get_model_info, inputs=[], outputs=[model_name_display, device_display, vram_display, dtype_display, config_display])

    ui_e["load_button"].click(
    _ui_load_model,
    inputs=[model_dropdown, custom_model_box, quant_dropdown, attn_dropdown],
    outputs=[ui_e["status_output"], model_name_display, device_display, vram_display, dtype_display, config_display],
    )
    gr.Blocks.load(
        iface,
        get_model_info,
        inputs=[],
        outputs=[model_name_display, device_display, vram_display, dtype_display, config_display]
    )


iface.launch(share=False, theme=gr.themes.Base(), css=css)