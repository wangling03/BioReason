import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    PreTrainedModel,
    PretrainedConfig,
)

from typing import Optional, List

from bioreason.models.dl.configuration_dl import DLConfig, DLDNAConfig

class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)

# BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}

class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        # block_layer = BLOCK_MAPPING[config.block_type]
        # self.model = ResNet(
        #     block_layer,
        #     config.layers,
        #     num_classes=config.num_classes,
        #     in_chans=config.input_channels,
        #     cardinality=config.cardinality,
        #     base_width=config.base_width,
        #     stem_width=config.stem_width,
        #     stem_type=config.stem_type,
        #     avg_down=config.avg_down,
        # )

    def forward(self, tensor):
        return self.model.forward_features(tensor)

class DLForConditionalGeneration(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        dna_model_name: str,
        cache_dir: str = None,
        max_length_dna: int = 1000,
        max_length_text: int = 512,
        text_model_finetune: bool = True,
        dna_model_finetune: bool = True,
        debug: bool = False,
    ):
        """
        Initialize the DNALLMModel.

        Args:
            text_model_name (str): Name of the text model to be used.
            dna_model_name (str): Name of the DNA model to be used. Defaults to nucleotide-transformer.
            cache_dir (str): Directory to cache the models.
            max_length_dna (int, optional): Maximum length of DNA sequences. Defaults to 1000.
            max_length_text (int, optional): Maximum length of text sequences. Defaults to 512.
            text_model_finetune (bool): Whether to finetune the text model. Defaults to True.
            dna_model_finetune (bool): Whether to finetune the DNA model. Defaults to True.
            debug (bool): Enable debug logging. Defaults to False.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.dna_model_finetune = dna_model_finetune
        self.debug = debug

        # Load the text model and tokenizer
        self.max_length_dna = max_length_dna
        self.max_length_text = max_length_text
        self.text = AutoModelForCausalLM.from_pretrained(
            text_model_name, cache_dir=cache_dir, trust_remote_code=True
        ).to("cuda")
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name, trust_remote_code=True
        )
        self.text_config = self.text.config

        # Load the DNA model and tokenizer
        self.dna_model = AutoModelForMaskedLM.from_pretrained(
            dna_model_name, cache_dir=cache_dir, trust_remote_code=True
        ).to("cuda")
        self.dna_tokenizer = AutoTokenizer.from_pretrained(
            dna_model_name, trust_remote_code=True
        )
        self.dna_config = self.dna_model.config

        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        self.text_hidden_size = self.text_config.hidden_size
        self.dna_hidden_size = self.dna_config.hidden_size

        # Create projection layer for DNA embeddings if needed
        if self.dna_hidden_size != self.text_hidden_size:
            self.dna_projection = nn.Linear(self.dna_hidden_size, self.text_hidden_size)
        else:
            self.dna_projection = nn.Identity()
        self.dna_projection = self.dna_projection.to("cuda")

    def get_dna_embeddings_batch(
        self, batch_dna_sequences: List[List[str]]
    ) -> List[torch.Tensor]:
        """
        Optimized version that processes DNA sequences more efficiently.

        Args:
            batch_dna_sequences: List of lists of DNA sequences per batch item

        Returns:
            List of tensor embeddings for each batch item
        """
        # Create a mapping to track which sequences belong to which batch item
        batch_idx_map = []
        all_sequences = []

        # Flatten all sequences with batch tracking
        for batch_idx, dna_sequences in enumerate(batch_dna_sequences):
            for seq in dna_sequences:
                all_sequences.append(seq)
                batch_idx_map.append(batch_idx)

        # If no sequences in the entire batch, return empty embeddings for each batch item
        if not all_sequences:
            return [
                torch.zeros((0, self.text_hidden_size), device="cuda")
                for _ in range(len(batch_dna_sequences))
            ]

        # Tokenize all sequences at once
        tokenized = self.dna_tokenizer(
            all_sequences,
            padding=True,
            truncation=True,
            max_length=self.max_length_dna,
            return_tensors="pt",
            return_attention_mask=True,
        ).to("cuda")

        # Process all sequences in a single forward pass
        with torch.no_grad():
            outputs = self.dna_model(
                input_ids=tokenized.input_ids,
                attention_mask=tokenized.attention_mask,
                output_hidden_states=True,
            )
            # Get the last hidden state
            hidden_states = outputs.hidden_states[
                -1
            ]  # shape: [n_seqs, seq_len, hidden_dim]

        # Project all embeddings at once if needed
        projected_states = self.dna_projection(hidden_states)

        # Group embeddings by batch item
        batch_size = len(batch_dna_sequences)
        result = [[] for _ in range(batch_size)]

        # For each sequence, get its embeddings and add to appropriate batch result
        for seq_idx, batch_idx in enumerate(batch_idx_map):
            # Get only the valid (non-padding) tokens
            valid_length = tokenized.attention_mask[seq_idx].sum().item()
            seq_embedding = projected_states[seq_idx, :valid_length]
            result[batch_idx].append(seq_embedding)

        # Concatenate embeddings for each batch item
        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                result[i] = torch.zeros((0, self.text_hidden_size), device="cuda")

        return result

    def forward(
        self,
        input_texts: Optional[List[str]] = None,
        batch_dna_sequences: Optional[List[List[str]]] = None,
        labels: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_texts (List[str]): List of input texts.
            batch_dna_sequences (List[List[str]]): List of lists of untokenized DNA sequences.
            labels (torch.Tensor): Labels for the model.
            input_ids (torch.Tensor): Input IDs for the text model.
            attention_mask (torch.Tensor): Attention mask for the text model.
            **kwargs: Additional arguments for the text model.
        
        """

        # If pre-tokenized inputs are provided, use them; otherwise tokenize the input_texts.
        if input_ids is None or attention_mask is None:
            tokenized_input = self.text_tokenizer(
                input_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length_text,
                return_tensors="pt",
            )
            input_ids = tokenized_input.input_ids.to("cuda")
            attention_mask = tokenized_input.attention_mask.to("cuda")
        else:
            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")

        batch_size = input_ids.shape[0]

        text_inputs_embeds = self.text.get_input_embeddings()(input_ids)

        # Handle batch_dna_sequences input
        if batch_dna_sequences is None:
            # No DNA sequences provided
            batch_dna_sequences = [[] for _ in range(batch_size)]
        elif isinstance(batch_dna_sequences[0], str):
            # Single example with multiple sequences provided as flat list
            batch_dna_sequences = [batch_dna_sequences]

        # Ensure batch_dna_sequences has the same batch size as input_ids
        if len(batch_dna_sequences) != batch_size:
            # Extend or truncate to match
            if len(batch_dna_sequences) < batch_size:
                batch_dna_sequences.extend(
                    [[] for _ in range(batch_size - len(batch_dna_sequences))]
                )
            else:
                batch_dna_sequences = batch_dna_sequences[:batch_size]

        # Get DNA embeddings for all batch items efficiently
        batch_dna_embeds = self.get_dna_embeddings_batch(batch_dna_sequences)

        # Initialize tensors to hold combined embeddings and attention masks
        max_dna_length = max((embed.shape[0] for embed in batch_dna_embeds), default=0)
        text_length = text_inputs_embeds.shape[1]
        combined_length = max_dna_length + text_length

        # Prepare tensors for batched operations
        combined_embeds_batch = torch.zeros(
            (batch_size, combined_length, self.text_hidden_size),
            device=text_inputs_embeds.device,
        )
        extended_attention_mask_batch = torch.zeros(
            (batch_size, combined_length),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        # Process all examples at once
        for i in range(batch_size):
            dna_embeds = batch_dna_embeds[i]
            num_dna_tokens = dna_embeds.shape[0]

            # Add DNA embeddings if any
            if num_dna_tokens > 0:
                combined_embeds_batch[i, :num_dna_tokens] = dna_embeds
                extended_attention_mask_batch[i, :num_dna_tokens] = 1

                if self.debug and i == 0:  # Debug only first example to avoid clutter
                    print(f"DNA embedding shape for example {i}: {dna_embeds.shape}")
                    print(f"Number of DNA tokens: {num_dna_tokens}")
                    # Print a few values to confirm they're not all zeros or NaNs
                    print(f"Sample DNA embedding values: {dna_embeds[0, :5]}")

            # Add text embeddings
            combined_embeds_batch[i, max_dna_length : max_dna_length + text_length] = (
                text_inputs_embeds[i]
            )
            extended_attention_mask_batch[
                i, max_dna_length : max_dna_length + text_length
            ] = attention_mask[i]

        # Trim to actual used length
        used_length = max_dna_length + text_length
        combined_embeds_batch = combined_embeds_batch[:, :used_length]
        extended_attention_mask_batch = extended_attention_mask_batch[:, :used_length]

        # Handle labels if provided
        if labels is not None:
            # Adjust labels to account for DNA tokens
            if labels.shape[1] != used_length:
                # Create a tensor filled with -100 (ignore_index for loss calculation)
                # This ensures DNA tokens and padding won't contribute to the loss
                adjusted_labels = torch.full(
                    (batch_size, used_length),
                    -100,  # Ignore index for loss calculation
                    dtype=labels.dtype,
                    device=labels.device,
                )

                # Copy label values at correct positions (after DNA tokens)
                # DNA tokens at the beginning (0:max_dna_length) will keep -100 values
                # ensuring they don't contribute to loss calculation
                text_start = max_dna_length
                for i in range(batch_size):
                    text_labels = labels[i]
                    valid_labels = text_labels != -100

                    if valid_labels.any():
                        try:
                            # Find first and last valid label position
                            valid_positions = valid_labels.nonzero(as_tuple=True)[0]

                            if len(valid_positions) > 0:
                                first_valid = valid_positions[0].item()
                                last_valid = valid_positions[-1].item() + 1

                                # Calculate valid ranges and ensure they're within bounds
                                src_start = first_valid
                                src_end = min(last_valid, text_labels.size(0))
                                tgt_start = text_start + first_valid
                                tgt_end = text_start + (src_end - src_start)

                                # Safety check for slice sizes
                                slice_size = src_end - src_start
                                if slice_size > 0:
                                    # Get the valid source and target ranges
                                    src_slice = text_labels[src_start:src_end]

                                    # Copy values directly without reshaping
                                    for j in range(slice_size):
                                        adjusted_labels[i, tgt_start + j] = src_slice[j]
                                elif self.debug:
                                    print(
                                        f"Skipping empty slice: src_start={src_start}, src_end={src_end}"
                                    )
                            elif self.debug:
                                print(f"No valid positions found in valid_positions")
                        except Exception as e:
                            if self.debug:
                                print(
                                    f"Error processing labels for batch item {i}: {e}"
                                )
                                print(f"text_labels shape: {text_labels.shape}")
                                print(
                                    f"valid_labels nonzero count: {valid_labels.sum().item()}"
                                )
                    elif self.debug:
                        print(f"No valid labels found for batch item {i}")

                # Ensure labels have the right shape
                if labels.shape[1] != used_length:
                    labels = adjusted_labels

                # Verify shapes match
                assert (
                    labels.shape == combined_embeds_batch.shape[:2]
                ), f"Label shape {labels.shape} doesn't match input shape {combined_embeds_batch.shape[:2]}"

        # Forward pass through the text model (loss is computed if labels is provided)
        outputs = self.text(
            inputs_embeds=combined_embeds_batch,
            attention_mask=extended_attention_mask_batch,
            labels=labels,
            **kwargs,
        )

        return outputs

    def generate(
        self,
        input_texts: List[str],
        batch_dna_sequences: List[List[str]],
        max_length: int = 100,
        **generation_kwargs,
    ) -> List[str]:

        batch_size = len(input_texts)

        # Handle batch_dna_sequences for generation
        if isinstance(batch_dna_sequences[0], str):
            # Single example with multiple sequences provided as flat list
            batch_dna_sequences = [batch_dna_sequences]

        assert batch_size == len(
            batch_dna_sequences
        ), "Batch size mismatch between input_texts and batch_dna_sequences"

        # Tokenize all input texts at once
        tokenized_input = self.text_tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length_text,
            return_tensors="pt",
        )
        input_ids = tokenized_input.input_ids.to("cuda")
        attention_mask = tokenized_input.attention_mask.to("cuda")

        # Get text embeddings
        text_inputs_embeds = self.text.get_input_embeddings()(input_ids)

        # Get batch of DNA embeddings efficiently
        batch_dna_embeds = self.get_dna_embeddings_batch(batch_dna_sequences)

        # Prepare combined embeddings and attention masks
        max_dna_length = max((embed.shape[0] for embed in batch_dna_embeds), default=0)
        text_length = text_inputs_embeds.shape[1]
        combined_length = max_dna_length + text_length

        combined_embeds_batch = torch.zeros(
            (batch_size, combined_length, self.text_hidden_size),
            device=text_inputs_embeds.device,
        )
        extended_attention_mask_batch = torch.zeros(
            (batch_size, combined_length),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )

        # Fill the tensors with embeddings and attention values
        for i in range(batch_size):
            dna_embeds = batch_dna_embeds[i]
            num_dna_tokens = dna_embeds.shape[0]

            if num_dna_tokens > 0:
                combined_embeds_batch[i, :num_dna_tokens] = dna_embeds
                extended_attention_mask_batch[i, :num_dna_tokens] = 1

            combined_embeds_batch[i, max_dna_length : max_dna_length + text_length] = (
                text_inputs_embeds[i]
            )
            extended_attention_mask_batch[
                i, max_dna_length : max_dna_length + text_length
            ] = attention_mask[i]

        # Trim to actual used length
        combined_embeds_batch = combined_embeds_batch[:, :combined_length]
        extended_attention_mask_batch = extended_attention_mask_batch[
            :, :combined_length
        ]

        # Generate text with the combined embeddings
        with torch.no_grad():
            outputs = self.text.generate(
                inputs_embeds=combined_embeds_batch,
                attention_mask=extended_attention_mask_batch,
                max_length=max_length,
                **generation_kwargs,
            )

        decoded_outputs = self.text_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return decoded_outputs
