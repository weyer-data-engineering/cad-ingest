"""Profiles: the vocabulary and the evidence of a kind of drawing, as data."""
import pytest

from app.plan2cad.features import Features
from app.plan2cad.profiles import ProfileError, installed, load


def profile(kind: str):
    return next(p for p in installed() if p.drawing == kind)


def test_the_built_in_profiles():
    assert {p.drawing for p in installed()} >= {"building", "electrical", "pid", "site"}


@pytest.mark.parametrize("kind, read, meant", [
    ("electrical", "-KI", "-K1"), ("electrical", "-F2 -SB", "-F2 -S8"), ("pid", "DNSO", "DN50"),
    ("pid", "PNI6", "PN16"), ("pid", "Pumpe", "Pumpe"), ("electrical", "-K12A", "-K12A"),  # no word of it: as read
])
def test_tokens_are_read_in_the_vocabulary_of_the_kind(kind, read, meant):
    assert profile(kind).read(read) == meant


def test_a_profile_that_is_not_one(tmp_path):
    (tmp_path / "bad.toml").write_text('format = 1\nid = "x"\n')
    with pytest.raises(ProfileError, match="drawing"):
        load(tmp_path / "bad.toml")
    (tmp_path / "bad.toml").write_text('format = 1\nid = "x"\ndrawing = "x"\n[vocabulary]\ntag = ["("]\n')
    with pytest.raises(ProfileError):
        load(tmp_path / "bad.toml")


def test_features_are_switched_by_name():
    assert not Features.of(off=["freehand"]).freehand and Features.of(on=["read_handwriting"]).read_handwriting
    with pytest.raises(ValueError, match="unknown"):
        Features.of(off=["nonsense"])
    with pytest.raises(ValueError, match="both"):
        Features.of(on=["dedupe"], off=["dedupe"])
