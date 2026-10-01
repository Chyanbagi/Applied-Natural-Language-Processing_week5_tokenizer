from tokenizers import Tokenizer

tokenizer_from_file = Tokenizer.from_file("korean_textbooks_tokenizer.json")

text = "지금은 응용자연어처리 수업 중입니다."
encoded = tokenizer_from_file.encode(text)

print("토큰화 결과:", encoded.tokens)
print("토큰 ID:", encoded.ids)