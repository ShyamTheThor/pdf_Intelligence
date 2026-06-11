from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging

from src.config import LLM_MODEL, LLM_DEVICE, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE

logging.getLogger("transformers").setLevel(logging.ERROR)


class QAPipeline:
    def __init__(self, model_name: str = LLM_MODEL):
        self.device = 0 if LLM_DEVICE == "cuda" and torch.cuda.is_available() else -1
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype=torch.float16 if self.device == 0 else torch.float32,
            device_map="auto" if self.device == 0 else None,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

    def generate(self, prompt: str) -> str:
        result = self.pipe(
            prompt,
            max_new_tokens=LLM_MAX_NEW_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=True,
        )
        return result[0]["generated_text"][len(prompt):].strip()

    def build_prompt(self, query: str, context: list[dict]) -> str:
        context_text = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in context
        )
        return (
            "<|im_start|>system\nYou are a helpful document assistant. Answer the question "
            "based only on the provided context. If the context doesn't contain the answer, "
            "say 'I cannot find this information in the provided documents.'<|im_end|>\n"
            f"<|im_start|>user\nContext:\n{context_text}\n\nQuestion: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
