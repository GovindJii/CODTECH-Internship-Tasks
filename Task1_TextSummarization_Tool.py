def simple_summarize(text, num_sentences=2):
    # 1. Basic cleaning and splitting into sentences
    sentences = text.split('. ')
    # 2. Simple word frequency (ignoring very common short words)
    stop_words = ["the", "and", "is", "of", "to", "in", "a", "i", "it", "that", "on", "for", "as", "with", "was"]
    words = text.lower().split()
    freq_table = {}
    for word in words:
        word = word.strip('.,!')
        if word not in stop_words and len(word) > 3:
            freq_table[word] = freq_table.get(word, 0) + 1
    # 3. Score sentences based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq
    # 4. Sort and pick the top sentences
    summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    return ". ".join(summarized_sentences) + "."
 
# --- Test Data ---
input_article = """."""
 
print("--- ORIGINAL ARTICLE ---")
print(input_article)
print("\n--- CONCISE SUMMARY ---")
print(simple_summarize(input_article, 2))