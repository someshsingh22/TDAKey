from flair.data import Sentence


def truncate_sequence(text, tokenizer, max_seq_len):
    """
    Truncate Longer sequences to max_seq_len

    text: Flair Sentence to be checked
    tokenizer: AutoTokenizer to be used
    max_seq_len: Maximum Length the model can handle
    """

    # Adding space for [CLS] and [SEP]
    max_seq_len -= 2

    tokenized_text = tokenizer.tokenize(text.to_plain_string())
    if len(tokenized_text) <= max_seq_len:
        return text
    else:
        limit = binary_search(text, tokenizer, max_seq_len)
        sent = " ".join(text.to_tokenized_string().split(" ")[:limit])
        truncated_sent = Sentence(tokenizer.tokenize(sent))
        for index in range(limit):
            truncated_sent[index].add_tag("ner", text[index].get_tag("ner").value)
        return text


def binary_search(text, tokenizer, max_seq_len) -> int:
    """
    Binary Search for the least words that can be encoded to 512 tokens after encoding

    text: Flair Sentence to be checked
    tokenizer: AutoTokenizer to be used
    max_seq_len: Maximum Length the model can handle
    """

    def check(limit) -> bool:
        """
        Internal method to check for BS condition

        limit: limit of words that can be encoded
        """
        reduced_tokens = text.to_tokenized_string().split(" ")[:limit]
        count = len(tokenizer.tokenize(" ".join(reduced_tokens)))
        return count > max_seq_len

    left, right = 1, max_seq_len
    while left < right:
        mid = left + (right - left) // 2
        if check(mid):
            right = mid
        else:
            left = mid + 1
    return left - 1
