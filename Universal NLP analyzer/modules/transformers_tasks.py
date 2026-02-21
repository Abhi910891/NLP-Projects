def transformer_tasks(text):
    try:
        from transformers import pipeline

        sentiment = pipeline("sentiment-analysis")
        result = sentiment(text)

        return result

    except Exception as e:
        return {"error": str(e), "engine": "disabled"}