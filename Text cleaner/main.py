from cleaner import clean_text

raw_text = """
Natural Language Processing (NLP) is AMAZING!!!
It helps computers understand human language.
Running, runs, ran — all should become run.
"""

output = clean_text(raw_text)

print("RAW TEXT:\n", raw_text)
print("\nCLEANED TEXT:\n", output)