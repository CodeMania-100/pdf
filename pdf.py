from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("yam-peleg/Hebrew-Mistral-7B")
model = AutoModelForCausalLM.from_pretrained("yam-peleg/Hebrew-Mistral-7B")

input_text = "תכתוב סיפור ילדים"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, min_length = 150, max_length = 200)
print(tokenizer.decode(outputs[0]))
