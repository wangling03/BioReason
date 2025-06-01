import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

from typing import Optional, List, Dict, Any, Union, Tuple

from bioreason.utils.dna_utils import DNAInput
from bioreason.models.dl.processing_dl import DLProcessor
from bioreason.models.dl.chat_template_dl import CHAT_TEMPLATE
from bioreason.models.evo2_tokenizer import Evo2Tokenizer

# Chat templates for different model types
qwen_chat_template = "{% set dna_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'dna' or 'dna' in content %}{% set dna_count.value = dna_count.value + 1 %}{% if add_vision_id %}Picture {{ dna_count.value }}: {% endif %}<|dna_start|><|dna_pad|><|dna_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

smol_template_old = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

past = "\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    "

past = "\n        {{- '<|im_start|>' + message.role + '\\n' }} {% if message.content is string %}{{ message.content }}<|im_end|>\n{% else %}{% for content in message.content %}{% if content.type == 'dna' or 'dna' in content %}{% set dna_count.value = dna_count.value + 1 %}{% if add_dna_id %}DNA Sequence {{ dna_count.value }}: {% endif %}<|dna_start|><|dna_pad|><|dna_end|>{% elif 'text' in content %}{{ content.text }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}"

qwen3_chat_template = "{%- set dna_count = namespace(value=0) %}{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and not(message.content is string and message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' }} {%- if message.content is string %}{{- message.content + '<|im_end|>' + '\\n' }}{%- else %}{%- for content in message.content %}{%- if content.type == 'dna' or 'dna' in content %}{%- set dna_count.value = dna_count.value + 1 %}{%- if add_dna_id %}DNA Sequence {{- dna_count.value }}: {%- endif %}<|dna_start|><|dna_pad|><|dna_end|>{%- elif 'text' in content %}{{- content.text }}{%- endif %}{%- endfor %}{{- '<|im_end|>' + '\\n' }}{%- endif %}{%- elif message.role == \"assistant\" %}\n        {%- set content = message.content[0].text %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is defined and message.reasoning_content is not none %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in message.content %}\n                {%- set content = message.content[0].text.split('</think>')[-1].lstrip('\\n') %}\n                {%- set reasoning_content = message.content[0].text.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if loop.index0 > ns.last_query_index %}\n            {%- if loop.last or (not loop.last and reasoning_content) %}\n                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n            {%- else %}\n                {{- '<|im_start|>' + message.role + '\\n' + content }}\n            {%- endif %}\n        {%- else %}\n            {{- '<|im_start|>' + message.role + '\\n' + content }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"


class DNALLMModel(nn.Module):
    """
    A combined model that processes both DNA sequences and text inputs.

    The model uses a DNA encoder (like NucleotideTransformer) to extract features from DNA sequences
    and a text model (LLM) to process text inputs and generate responses. The DNA features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        text_model_name: str,
        dna_model_name: str,
        cache_dir: Optional[str] = None,
        max_length_dna: int = 2048,
        max_length_text: int = 512,
        text_model_finetune: bool = True,
        dna_model_finetune: bool = True,
        dna_is_evo2: bool = False,
        dna_embedding_layer: str = None
    ):
        """
        Initialize the DNALLMModel.

        Args:
            text_model_name: Name of the text model to be used.
            dna_model_name: Name of the DNA model to be used.
            cache_dir: Directory to cache the models.
            max_length_dna: Maximum length of DNA sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 512.
            text_model_finetune: Whether to finetune the text model. Defaults to True.
            dna_model_finetune: Whether to finetune the DNA model. Defaults to True.
            dna_is_evo2: Whether the DNA model is Evo2. Defaults to False.
            dna_embedding_layer: Name of the layer to use for the Evo2 model. Defaults to None.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.dna_model_finetune = dna_model_finetune
        self.max_length_dna = max_length_dna
        self.max_length_text = max_length_text
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer


        # Load the text model and tokenizer
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, trust_remote_code=True)
        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = qwen3_chat_template
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.dna_token_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")


        # Load the DNA model and tokenizer
        if not self.dna_is_evo2:
            self.dna_model = AutoModelForMaskedLM.from_pretrained(
                dna_model_name, cache_dir=cache_dir, trust_remote_code=True
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=True)
            self.dna_config = self.dna_model.config

        else:
            from evo2 import Evo2
            self.dna_model = Evo2(dna_model_name)
            self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
            self.dna_config = self.dna_model.model.config
            self.dna_embedding_layer = self.dna_embedding_layer

        # Get model dimensions
        self.text_hidden_size = self.text_config.hidden_size
        self.dna_hidden_size = self.dna_config.hidden_size

        # Create projection layer to map DNA embeddings to text model's embedding space
        self.dna_projection = nn.Linear(self.dna_hidden_size, self.text_hidden_size)

        # Create processor for handling inputs
        self.processor = DLProcessor(tokenizer=self.text_tokenizer, dna_tokenizer=self.dna_tokenizer)

    
    def process_dna_embeddings(
        self,
        dna_tokenized: Dict[str, torch.Tensor],
        batch_idx_map: List[int],
        batch_size: int,
    ) -> List[torch.Tensor]:
        """
        Process DNA sequences to obtain embeddings.

        Args:
            dna_tokenized: Tokenized DNA sequences
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch

        Returns:
            List of tensor embeddings for each batch item
        """
        # Process all sequences to get DNA representations
        with torch.no_grad():
            # Handle different model types based on dna_is_evo2 attribute
            if self.dna_is_evo2 and self.dna_embedding_layer is not None:  # Evo2 model
                # Get embeddings from the specific layer in Evo2
                hidden_states_list = []
                
                for seq_idx in range(len(dna_tokenized["input_ids"])):
                    # Extract single sequence
                    input_ids = dna_tokenized["input_ids"][seq_idx:seq_idx+1]
                    
                    # Call Evo2 with return_embeddings=True
                    _, embeddings = self.dna_model(
                        input_ids,
                        return_embeddings=True,
                        layer_names=[self.dna_embedding_layer]
                    )
                    
                    # Get embeddings for the specified layer
                    seq_embeddings = embeddings[self.dna_embedding_layer].squeeze(0)
                    hidden_states_list.append(seq_embeddings)
                
                # Stack to get same format as non-Evo2 output
                if hidden_states_list:
                    hidden_states = torch.stack(hidden_states_list)
                else:
                    return [torch.zeros((0, self.text_hidden_size)) for _ in range(batch_size)]
                    
            else:  # Standard HuggingFace model
                # Use existing code path for HF models
                outputs = self.dna_model(
                    input_ids=dna_tokenized["input_ids"],
                    attention_mask=dna_tokenized["attention_mask"],
                    output_hidden_states=True,
                )
                # Get the last hidden state
                hidden_states = outputs.hidden_states[-1]  # shape: [n_seqs, seq_len, hidden_dim]

        # Project all embeddings at once
        hidden_states = hidden_states.to(device=self.dna_projection.weight.device, dtype=self.dna_projection.weight.dtype)
        projected_states = self.dna_projection(hidden_states)

        # Group embeddings by batch item
        result = [[] for _ in range(batch_size)]

        # For each sequence, get its embeddings and add to appropriate batch result
        for seq_idx, batch_idx in enumerate(batch_idx_map):
            # Get only the valid (non-padding) tokens
            valid_length = dna_tokenized["attention_mask"][seq_idx].sum().item()
            seq_embedding = projected_states[seq_idx, :valid_length]
            result[batch_idx].append(seq_embedding)

        # Concatenate embeddings for each batch item
        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                result[i] = torch.zeros((0, self.text_hidden_size))

        return result

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text based on DNA and text inputs.

        Args:
            input_ids: Input IDs (used if provided directly)
            attention_mask: Attention mask (used if provided directly)
            dna_tokenized: Tokenized DNA sequences (used if provided directly)
            batch_idx_map: Batch mapping for DNA sequences (used if provided directly)
            labels: Labels for supervised fine-tuning (used if provided directly)
            **kwargs: Additional arguments for generation

        Returns:
            Outputs from the text model
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("Either 'inputs' or 'input_ids'/'attention_mask' must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if dna_tokenized is not None and batch_idx_map:
            batch_dna_embeds = self.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)

            mask = input_ids == self.dna_token_id

            n_dna_tokens = mask.sum().item()
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)
            n_dna_features = dna_embeds_flat.shape[0]

            if n_dna_features != n_dna_tokens:
                raise ValueError(
                    f"DNA features and DNA tokens do not match: features {n_dna_features}, tokens: {n_dna_tokens}"
                )

            # Ensure DNA embeddings have the same dtype as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype)
            text_inputs_embeds[mask] = dna_embeds_flat

        # Handle labels if provided (for training)
        if labels is not None:
            # TODO: Implement this
            pass

        # Forward pass through the text model (loss is computed if labels is provided)
        outputs = self.text_model(
            inputs_embeds=text_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on DNA and text inputs.

        Args:
            inputs: The preprocessed inputs from the processor (preferred method)
            batch_dna_sequences: List of lists of DNA sequences per batch item (legacy method)
            input_texts: List of input texts (legacy method)
            input_ids: Input IDs (used if provided directly)
            attention_mask: Attention mask (used if provided directly)
            dna_tokenized: Tokenized DNA sequences (used if provided directly)
            batch_idx_map: Batch mapping for DNA sequences (used if provided directly)
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs which can be decoded using the processor
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("Either 'inputs' or 'input_ids'/'attention_mask' must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if dna_tokenized is not None and batch_idx_map:
            batch_dna_embeds = self.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size)

            mask = input_ids == self.dna_token_id

            n_dna_tokens = mask.sum().item()
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)
            n_dna_features = dna_embeds_flat.shape[0]

            if n_dna_features != n_dna_tokens:
                raise ValueError(
                    f"DNA features and DNA tokens do not match: features {n_dna_features}, tokens: {n_dna_tokens}"
                )

            # Ensure DNA embeddings have the same dtype as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype)
            text_inputs_embeds[mask] = dna_embeds_flat

        # Generation parameters may need adjustment based on model type
        with torch.no_grad():
            outputs = self.text_model.generate(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                **generation_kwargs,
            )

        return outputs