def load_data(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        words, tags = [], []
        for line in f:
            line = line.strip()
            
            # Handle sentence boundaries
            if not line:
                if words:  # Skip empty sequences
                    data.append((words, tags))
                    words, tags = [], []
                continue
            
            # Split word and tag
            word, tag = line.split('\t')  # Adjust separator if needed
            words.append(word)
            tags.append(tag)
            
        # Handle last sequence
        if words:
            data.append((words, tags))
    
    return data