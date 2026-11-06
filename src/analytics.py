def build_metrics(tickets):
    return {
        "total": int(len(tickets)),
        "high_priority": int((tickets["priority"] == "high").sum()),
        "negative": int((tickets["sentiment"] == "negative").sum()),
    }
