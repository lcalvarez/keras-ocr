import os
import time
from huggingface_hub import hf_hub_download
from utils import check_connectivity, toggle_wifi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM


def main():
    HUGGING_FACE_API_KEY = ""

    model_id = "lmsys/fastchat-t5-3b-v1.0"
    filenames = [
            "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
            "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
    ]

    for filename in filenames:
            downloaded_model_path = hf_hub_download(
                        repo_id=model_id,
                        filename=filename,
                        token=HUGGING_FACE_API_KEY
            )
            print(downloaded_model_path)

#    print(check_connectivity())
#    toggle_wifi("off")
#    time.sleep(0.5)
#    print(check_connectivity())
#    toggle_wifi("on")

    tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    llm = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)

    llm("What are competitors to Apache Kafka?")

    llm("""My name is Mark.
    I have brothers called David and John and my best friend is Michael.
    Using only the context above. Do you know if I have a sister?    
    """)

    return llm