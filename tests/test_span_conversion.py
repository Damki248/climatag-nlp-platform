from training.ner_gliner.train import parse_climate_model_annotations
import json, tempfile

def _task(text, start, end, label="Climate Model"):
    return {"data": {"text": text},
            "annotations": [{"result": [{"value": {"start": start, "end": end,
                            "text": text[start:end], "labels": [label]}}]}]}

def _parse(tasks):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(tasks, f)
        path = f.name
    return parse_climate_model_annotations(path)

def test_aligned_span():
    s = _parse([_task("The CMIP6 model works", 4, 9)])
    assert s[0]["ner"] == [(1, 2, "Climate Model")]

def test_span_inside_token_with_parens():
    s = _parse([_task("The (CMIP6) model works", 5, 10)])
    assert s[0]["ner"] == [(1, 2, "Climate Model")]  # proširi na "(CMIP6)"

def test_multiword_entity_with_space():
    s = _parse([_task("Under RCP 8.5 scenarios", 6, 13)])
    assert s[0]["ner"] == [(1, 3, "Climate Model")]  # "RCP" + "8.5"

def test_no_inverted_spans_in_real_data():
    real = json.load(open("data/annotations/climate_model_annotations.json"))
    for sample in _parse(real):
        for (ts, te, _) in sample["ner"]:
            assert ts < te <= len(sample["tokenized_text"])