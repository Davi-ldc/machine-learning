from transformers import GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large', pad_token_id=tokenizer.pad_token_id)


title = "love"

txt = tokenizer.encode(title, return_tensors='pt')
 
outputs = model.generate(txt, max_length=200, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))