from transformers import BertModel, BertConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch

# ------------------------------
# Step 1: Create and wrap BERT
# ------------------------------
config = BertConfig(
    model_type="bert",  # ✅ REQUIRED for saving and loading
    hidden_size=512,
    num_hidden_layers=3,
    num_attention_heads=8,
    intermediate_size=512 * 4,
    max_position_embeddings=512,
    vocab_size=30522,
)

bert = BertModel(config)

# Create and wrap with LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)
peft = get_peft_model(bert, lora_config)

# ------------------------------
# Step 2: Save base + lora model
# ------------------------------
save_path = "debug_model"
peft.save_pretrained(save_path)
peft.base_model.model.config.save_pretrained(save_path)
torch.save(peft.base_model.model.state_dict(), f"{save_path}/bert_model.pt")

# ------------------------------
# Step 3: Load base + lora model
# ------------------------------
config_loaded = BertConfig.from_pretrained(save_path, local_files_only=True)
bert2 = BertModel(config_loaded)
bert2.load_state_dict(torch.load(f"{save_path}/bert_model.pt"))

peft2 = PeftModel.from_pretrained(bert2, save_path, local_files_only=True)

# ------------------------------
# Step 4: Sanity check
# ------------------------------
x = torch.randn(1, 5, 512)
peft.eval()
peft2.eval()
out1 = peft(x).last_hidden_state
out2 = peft2(x).last_hidden_state
print("✅ Sanity check passed:", (out1 - out2).abs().max())
