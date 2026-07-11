from tests.helpers import make_unlearn_batch


def test_unlearn_batch_has_forget_and_retain_components():
    batch = make_unlearn_batch(batch_size=2, sequence_length=6)
    assert set(batch) == {"forget", "retain"}
    assert batch["forget"]["labels"].shape == (2, 6)
    assert (batch["retain"]["labels"][:, :2] == -100).all()
