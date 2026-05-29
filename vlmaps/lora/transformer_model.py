from __future__ import annotations
import os
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftMixedModel
from pathlib import Path
from copy import deepcopy


#  █████╗ ██████╗  ██████╗
# ██╔══██╗██╔══██╗██╔════╝
# ███████║██████╔╝██║
# ██╔══██║██╔══██╗██║
# ██║  ██║██████╔╝╚██████╗
# ╚═╝  ╚═╝╚═════╝  ╚═════╝


class TransformerModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x) -> torch.Tensor:
        pass

    @abstractmethod
    def save(self, save_path: str, force: bool = False) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, load_path: str, dim: int) -> TransformerModel:
        pass


# ███╗   ███╗ █████╗ ██╗███╗   ██╗     ██████╗██╗      █████╗ ███████╗███████╗
# ████╗ ████║██╔══██╗██║████╗  ██║    ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝
# ██╔████╔██║███████║██║██╔██╗ ██║    ██║     ██║     ███████║███████╗███████╗
# ██║╚██╔╝██║██╔══██║██║██║╚██╗██║    ██║     ██║     ██╔══██║╚════██║╚════██║
# ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║    ╚██████╗███████╗██║  ██║███████║███████║
# ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝     ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝


class PretrainedAggregator(TransformerModel):

    def __init__(
        self,
        dim: int = 512,
        max_position_embeddings: int = 512,
        lora_config: Optional[LoraConfig] = None,
    ) -> None:
        super().__init__()
        config = BertConfig(
            hidden_size=dim,
            num_hidden_layers=3,
            num_attention_heads=8,
            intermediate_size=dim * 4,
            max_position_embeddings=max_position_embeddings,
            vocab_size=30522,
        )
        base = BertModel(config)
        self.bert: Union[BertModel, PeftModel, PeftMixedModel] = (
            get_peft_model(base, lora_config) if lora_config else base
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        # tiny cache for masks/pos (device, T, dtype) -> tensors
        self._mask_cache: Dict[Tuple[str, int, str], torch.Tensor] = {}
        self._pos_cache: Dict[Tuple[str, int, str], torch.Tensor] = {}

    def forward(
        self,
        x=None,
        *,
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        if (x is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of (x) or (inputs_embeds).")
        tokens = x if inputs_embeds is None else inputs_embeds  # (B,T,D)
        assert tokens is not None
        B, T, _ = tokens.shape
        device = tokens.device

        # prepend CLS
        cls = self.cls_token.expand(B, 1, -1)
        tokens_cat = torch.cat([cls, tokens], dim=1)  # (B, T+1, D)
        N = T + 1

        # ---------- FAST PATH ----------
        # if caller didn't pass masks/ids, let BERT create them internally.
        if position_ids is None and attention_mask is None:
            return F.normalize(
                self.proj(
                    self.bert(inputs_embeds=tokens_cat).last_hidden_state[:, 0, :]
                ),
                dim=-1,
            )

        # ---------- SLOW PATH (only when user provides masks/ids) ----------
        # reconcile attention_mask
        if attention_mask is None:
            attention_mask = self._cached_full_ones(device, B, N, torch.long)
        else:
            if attention_mask.size(1) == T:
                pad = self._cached_full_ones(device, B, 1, attention_mask.dtype)
                attention_mask = torch.cat([pad, attention_mask], dim=1)
            elif attention_mask.size(1) != N:
                raise ValueError(
                    f"attention_mask length {attention_mask.size(1)} != {N}"
                )

        # reconcile position_ids
        if position_ids is None:
            position_ids = self._cached_positions(device, B, N, torch.long)
        else:
            if position_ids.size(1) == T:
                first = self._cached_positions(
                    device, B, 1, position_ids.dtype
                )  # zeros
                position_ids = torch.cat([first, position_ids + 1], dim=1)
            elif position_ids.size(1) != N:
                raise ValueError(f"position_ids length {position_ids.size(1)} != {N}")

        # token_type_ids
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, N, dtype=torch.long, device=device)
        else:
            if token_type_ids.size(1) == T:
                pad = torch.zeros(B, 1, dtype=token_type_ids.dtype, device=device)
                token_type_ids = torch.cat([pad, token_type_ids], dim=1)
            elif token_type_ids.size(1) != N:
                raise ValueError(
                    f"token_type_ids length {token_type_ids.size(1)} != {N}"
                )

        out = self.bert(
            inputs_embeds=tokens_cat,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return F.normalize(self.proj(out.last_hidden_state[:, 0, :]), dim=-1)

    #  ██████╗ █████╗  ██████╗██╗  ██╗███████╗
    # ██╔════╝██╔══██╗██╔════╝██║  ██║██╔════╝
    # ██║     ███████║██║     ███████║█████╗
    # ██║     ██╔══██║██║     ██╔══██║██╔══╝
    # ╚██████╗██║  ██║╚██████╗██║  ██║███████╗
    #  ╚═════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚══════╝

    def _cached_full_ones(self, device, B, N, dtype):
        key = (device, N, dtype)
        t = self._mask_cache.get(key)
        if t is None or t.size(0) < B:
            # store a big-enough (max B so far) tensor; slice on use
            t = torch.ones(B, N, device=device, dtype=dtype)
            self._mask_cache[key] = t
        return t[:B]

    def _cached_positions(self, device, B, N, dtype):
        key = (device, N, dtype)
        t = self._pos_cache.get(key)
        if t is None or t.size(0) < B:
            base = torch.arange(N, device=device, dtype=dtype)
            t = base.unsqueeze(0).expand(B, N)
            self._pos_cache[key] = t
        return t[:B]

    # ███████╗ █████╗ ██╗   ██╗███████╗
    # ██╔════╝██╔══██╗██║   ██║██╔════╝
    # ███████╗███████║██║   ██║█████╗
    # ╚════██║██╔══██║╚██╗ ██╔╝██╔══╝
    # ███████║██║  ██║ ╚████╔╝ ███████╗
    # ╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝

    def save(self, save_path: str, force: bool = False) -> None:
        """
        Saves:
          - If LoRA: adapter files (adapter_config.json + adapter_model.safetensors)
          - Base BERT weights merged (to bert_model.pt) WITHOUT destroying adapters
          - cls_token.pt and proj.pt
        """

        def _check_overwrite(save_path: str, file: str):
            want_to_save = True
            if not force:
                if Path(f"{save_path}/{file}").exists():
                    want_to_save = False
                    choice = input("File exists, overwrite? y/n: ")
                    if choice.lower() == "y":
                        want_to_save = True
                    else:
                        choice2 = input("Input new path? y/n: ")
                        if choice2.lower() == "y":
                            save_path = input("Enter new path: ")
                            want_to_save = True
                        else:
                            want_to_save = False
            return save_path, want_to_save

        save_path, want_to_save = _check_overwrite(save_path, "bert_model.pt")
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        if want_to_save:
            # 1) Save adapters FIRST (so we don’t risk losing them if anything goes wrong)
            if isinstance(self.bert, PeftModel):
                self.bert.save_pretrained(str(save_dir))

                # 2) Save a merged base copy WITHOUT destroying adapters on the live model
                bert_copy: PeftModel = deepcopy(self.bert).to("cpu")
                # returns HF model with LoRA merged
                merged_base: BertModel = bert_copy.merge_and_unload()  # type: ignore
                torch.save(merged_base.state_dict(), f"{save_dir}/bert_model.pt")

            else:
                # Plain BERT: just save the base weights
                torch.save(self.bert.state_dict(), f"{save_dir}/bert_model.pt")

            # 3) Save the rest
            torch.save(self.cls_token.detach().cpu(), f"{save_dir}/cls_token.pt")
            torch.save(self.proj.state_dict(), f"{save_dir}/proj.pt")
            print(f"✅ Saved aggregator to {save_dir}")
        else:
            print("Not saved")

    # ██╗      ██████╗  █████╗ ██████╗
    # ██║     ██╔═══██╗██╔══██╗██╔══██╗
    # ██║     ██║   ██║███████║██║  ██║
    # ██║     ██║   ██║██╔══██║██║  ██║
    # ███████╗╚██████╔╝██║  ██║██████╔╝
    # ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝

    @classmethod
    def load(
        cls,
        load_path: str,
        dim: int = 512,
        max_tokens: int = 512,
        extend_pos_embeddings: int = -1,
    ) -> "PretrainedAggregator":
        """
        Loads base from bert_model.pt, optionally extends positions, then attaches LoRA adapters
        if an adapter_config.json is present.
        """
        load_dir = Path(load_path)
        if not load_dir.exists():
            raise FileNotFoundError(load_dir)

        # Recreate base config consistent with training (you can also save a config file if you prefer)
        config = BertConfig(
            hidden_size=dim,
            num_hidden_layers=3,
            num_attention_heads=8,
            intermediate_size=dim * 4,
            max_position_embeddings=max_tokens,
            vocab_size=30522,
        )

        base = BertModel(config)
        sd = torch.load(load_dir / "bert_model.pt", map_location="cpu")
        base.load_state_dict(sd, strict=True)

        # Optional: extend positional embeddings for longer N
        if (
            extend_pos_embeddings
            and extend_pos_embeddings > base.config.max_position_embeddings
        ):
            PretrainedAggregator.extend_bert_pos_emb(
                base, extend_pos_embeddings, init="copy_last"
            )

        # If adapters exist, attach them
        adapter_cfg = load_dir / "adapter_config.json"
        bert: Union[BertModel, PeftModel]
        if adapter_cfg.exists():
            bert = PeftModel.from_pretrained(
                model=base, model_id=str(load_dir), local_files_only=True
            )
        else:
            bert = base

        # Bypass __init__ to avoid re-initializing layers
        model = cls.__new__(cls)
        super(cls, model).__init__()
        model._mask_cache = {}
        model._pos_cache = {}
        model.bert = bert
        model.cls_token = nn.Parameter(
            torch.load(load_dir / "cls_token.pt", map_location="cpu")
        )
        model.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        model.proj.load_state_dict(torch.load(load_dir / "proj.pt", map_location="cpu"))

        return model

    # ██╗   ██╗████████╗██╗██╗     ███████╗
    # ██║   ██║╚══██╔══╝██║██║     ██╔════╝
    # ██║   ██║   ██║   ██║██║     ███████╗
    # ██║   ██║   ██║   ██║██║     ╚════██║
    # ╚██████╔╝   ██║   ██║███████╗███████║
    #  ╚═════╝    ╚═╝   ╚═╝╚══════╝╚══════╝

    @staticmethod
    def extend_bert_pos_emb(
        model: BertModel, new_max_len: int, init: str = "copy_last"
    ):
        """
        Safely extend BERT learned positional embeddings to new_max_len.
        """
        old_emb: nn.Embedding = model.embeddings.position_embeddings
        old_len, dim = old_emb.weight.shape
        if new_max_len <= old_len:
            model.config.max_position_embeddings = new_max_len
            return model

        new_emb = nn.Embedding(
            new_max_len, dim, device=old_emb.weight.device, dtype=old_emb.weight.dtype
        )
        with torch.no_grad():
            new_emb.weight[:old_len] = old_emb.weight
            if init == "copy_last":
                new_emb.weight[old_len:] = old_emb.weight[-1].expand(
                    new_max_len - old_len, -1
                )
            elif init == "zeros":
                new_emb.weight[old_len:] = 0
            elif init == "normal":
                new_emb.weight[old_len:].normal_(mean=0.0, std=0.02)
            elif init == "interpolate":
                w = old_emb.weight.T.unsqueeze(0).unsqueeze(0)  # (1,1,dim,old_len)
                w_interp = F.interpolate(
                    w, size=(dim, new_max_len), mode="bilinear", align_corners=True
                )
                new_emb.weight.copy_(w_interp.squeeze(0).squeeze(0).T)
            else:
                raise ValueError(f"Unknown init: {init}")

        model.embeddings.position_embeddings = new_emb
        model.config.max_position_embeddings = new_max_len
        print(f"➡️  Extended BERT positional embeddings to {new_max_len}")
        return model

    @staticmethod
    def bert_forward_with_embeds(model: PretrainedAggregator, embeds: torch.Tensor):
        """
        Forward helper when you already have embeddings (B, N, D).
        """
        B, N, _ = embeds.shape
        device = embeds.device
        position_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
        attention_mask = torch.ones(B, N, device=device, dtype=torch.long)
        token_type_ids = torch.zeros(B, N, dtype=torch.long, device=device)
        return model(
            inputs_embeds=embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    # ██╗      ██████╗ ██████╗  █████╗
    # ██║     ██╔═══██╗██╔══██╗██╔══██╗
    # ██║     ██║   ██║██████╔╝███████║
    # ██║     ██║   ██║██╔══██╗██╔══██║
    # ███████╗╚██████╔╝██║  ██║██║  ██║
    # ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝


class LoraAggregator(PretrainedAggregator):
    def __init__(self, dim: int = 512, max_position_embeddings: int = 512) -> None:
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        super().__init__(dim, max_position_embeddings, lora_config)

    @classmethod
    def load(
        cls,
        load_path: str,
        dim: int = 512,
        max_tokens: int = 512,
        extend_pos_embeddings: int = -1,
    ) -> PretrainedAggregator:
        return PretrainedAggregator.load(
            load_path, dim, max_tokens, extend_pos_embeddings
        )


# ██╗   ██╗ █████╗ ███╗   ██╗██╗██╗     ██╗      █████╗
# ██║   ██║██╔══██╗████╗  ██║██║██║     ██║     ██╔══██╗
# ██║   ██║███████║██╔██╗ ██║██║██║     ██║     ███████║
# ╚██╗ ██╔╝██╔══██║██║╚██╗██║██║██║     ██║     ██╔══██║
#  ╚████╔╝ ██║  ██║██║ ╚████║██║███████╗███████╗██║  ██║
#   ╚═══╝  ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝╚══════╝╚═╝  ╚═╝


class VectorTransformer(TransformerModel):
    def __init__(self, dim=512, heads=8, layers=3):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # (B, T, D)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.norm(x[:, 0])  # (B, D)

    def save(self, save_path: str, force: bool = False):
        file_path = f"{save_path}/aggregator_transformer.pt"

        want_to_save = True
        if not force:
            if Path(file_path).exists():
                want_to_save = False
                choice = input("File exists, overwrite? y/n: ")
                if choice.lower() == "y":
                    want_to_save = True
                else:
                    choice2 = input("Input new path? y/n: ")
                    if choice2.lower() == "y":
                        save_path = input("Enter new path: ")
                        file_path = f"{save_path}/aggregator_transformer.pt"
                        want_to_save = True
                    else:
                        want_to_save = False

        if want_to_save:
            Path(save_path).mkdir(exist_ok=True, parents=True)
            torch.save(
                {"model_state_dict": self.state_dict()},
                file_path,
            )
            print(f"Model saved: {file_path}")
        else:
            print("Model not saved")

    @classmethod
    def load(cls, load_path: str, dim: int = 512):
        print("📦 Loading saved VectorTransformer weights...")
        file_path = f"{load_path}/aggregator_transformer.pt"
        checkpoint = torch.load(file_path)
        model = cls(dim=dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
