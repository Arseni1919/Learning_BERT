# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#
# # Get PAD token string
# pad_token = tokenizer.pad_token
# print(f"PAD token: '{pad_token}'")  # Output: '[PAD]'
#
# # Get PAD token ID
# pad_token_id = tokenizer.pad_token_id
# print(f"PAD token ID: {pad_token_id}")  # Output: 0
#
# # Alternative way to get PAD token ID
# pad_id_alt = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# print(f"PAD token ID (alternative): {pad_id_alt}")  # Output: 0
import time

from tqdm import tqdm

for review in tqdm(range(10)):
    time.sleep(1)

