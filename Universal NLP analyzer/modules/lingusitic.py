def linguistic_analysis(text):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        pos = [(t.text, t.pos_) for t in doc]
        ner = [(e.text, e.label_) for e in doc.ents]
        dep = [(t.text, t.dep_, t.head.text) for t in doc]

        return {
            "pos": pos,
            "ner": ner,
            "dep": dep,
            "engine": "spacy"
        }

    except Exception:
        words = text.split()
        return {
            "pos": [(w, "WORD") for w in words],
            "ner": [],
            "dep": [],
            "engine": "fallback"
        }