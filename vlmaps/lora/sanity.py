from transformers import BertModel, BertConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch

# Define BERT config and inject model_type
config = BertConfig(
    hidden_size=512,
    num_hidden_layers=3,
    num_attention_heads=8,
    intermediate_size=512 * 4,
    max_position_embeddings=512,
    vocab_size=30522,
)

bert = BertModel(config)
torch.save(bert.state_dict(), "debug_model/bert_model.pt")

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)
peft = get_peft_model(bert, lora_config)
peft.save_pretrained("debug_model")

bert2 = BertModel(config)
bert2.load_state_dict(torch.load("debug_model/bert_model.pt"))
peft2 = PeftModel.from_pretrained(bert2, "debug_model")

print("✅ Loaded LoRA model successfully!")

device = "cuda"
test_input = torch.randn(1, 5, 512).to(device)
peft.eval()
peft2.eval()
out1 = bert(test_input)
out2 = bert2(test_input)

print("Diff:", (out1 - out2).abs().max())
