from evals import lm_eval as lm_eval_module


class DummyModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True


def test_prepare_model_reuses_trainer_tokenizer(monkeypatch):
    calls = []

    def fake_hflm(model, **kwargs):
        calls.append((model, kwargs))
        return "prepared"

    monkeypatch.setattr(lm_eval_module, "HFLM", fake_hflm)
    evaluator = object.__new__(lm_eval_module.LMEvalEvaluator)
    model = DummyModel()
    tokenizer = object()

    prepared = evaluator.prepare_model(model, tokenizer=tokenizer)

    assert prepared == "prepared"
    assert model.eval_called
    assert calls == [(model, {"tokenizer": tokenizer})]
