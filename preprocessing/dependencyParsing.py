import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Read input from file
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Process the text
doc = nlp(text)

# Write dependency parsing results to output file
with open('dependency_results.txt', 'w', encoding='utf-8') as output_file:
    for token in doc:
        output_file.write(f"{token.text} -> {token.head.text} ({token.dep_})\n")

print("Dependency parsing results have been written to 'dependency_results.txt'")