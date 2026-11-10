"""Profiles: the vocabulary and the evidence of a kind of drawing, as data."""
import pytest

from app.plan2cad.features import Features
from app.plan2cad.profiles import BY_KIND, ProfileError, installed, load


def profile(kind: str):
    return next(p for p in installed() if p.drawing == kind)


def test_the_built_in_profiles():
    assert {p.drawing for p in installed()} >= {"building", "electrical", "pid", "site"}


@pytest.mark.parametrize("kind, read, meant", [
    ("electrical", "-KI", "-K1"), ("electrical", "-F2 -SB", "-F2 -S8"), ("pid", "DNSO", "DN50"),
    ("pid", "PNI6", "PN16"), ("pid", "Pumpe", "Pumpe"), ("electrical", "-K12A", "-K12A"),  # no word of it: as read
    ("pid", "8-501 DN80-P-1O11", "B-501 DN80-P-1011"), ("pid", "1-101 0-100", "1-101 0-100"),  # no I or O tags
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


def test_a_framed_single_character_is_read_as_the_kind_frames_them(tmp_path):
    assert profile("building").alone("8") == profile("pid").alone("8") == "8"  # frames hold letters and digits alike
    (tmp_path / "x.toml").write_text('format = 1\nid = "x"\ndrawing = "x"\n[reading]\nframed = "letter"\n')
    assert [load(tmp_path / "x.toml").alone(s) for s in ("8", "1", "B", "4", "AB")] == ["B", "I", "B", "4", "AB"]
    (tmp_path / "x.toml").write_text('format = 1\nid = "x"\ndrawing = "x"\n[reading]\nframed = "digit"\n')
    assert load(tmp_path / "x.toml").alone("O") == "0"
    (tmp_path / "x.toml").write_text('format = 1\nid = "x"\ndrawing = "x"\n[reading]\nframed = "letters"\n')
    with pytest.raises(ProfileError, match="framed"):
        load(tmp_path / "x.toml")


def test_words_of_the_vocabulary_are_counted_as_read_or_with_confused_characters_swapped():
    assert profile("building").words("0,60") == 1 and profile("building").words("09'0") == 0
    assert profile("electrical").words("-KI -F2 Motor") == 2


def test_a_token_is_a_word_of_the_vocabulary_as_it_stands():
    assert profile("pid").word("DN80-P-1011") and profile("pid").word("B-501")
    assert not profile("pid").word("0N80-P-1011") and not profile("pid").word("8-501")


def test_a_kind_switches_steps_off_unless_the_job_switches_them_on(tmp_path):
    assert profile("pid").steps_off == profile("electrical").steps_off == ("dimensions",)
    assert profile("building").steps_off == profile("site").steps_off == ()  # dimensioned in chains
    assert Features().for_kind(("dimensions",)) == (Features.of(off=["dimensions"]), ["dimensions"])
    assert Features.of(on=["dimensions"]).for_kind(("dimensions",)) == (Features(), [])  # the job's on wins
    assert Features.of(off=["dimensions"]).for_kind(("dimensions",))[1] == []  # off already: not the kind's doing
    assert Features().for_kind(()) == (Features(), [])
    assert "asked" not in Features.names()
    for step in ("nonsense", "freehand"):  # no step, or one that runs before the kind is known
        (tmp_path / "x.toml").write_text(f'format = 1\nid = "x"\ndrawing = "x"\n[steps]\noff = ["{step}"]\n')
        with pytest.raises(ProfileError, match="steps.off"):
            load(tmp_path / "x.toml")
    assert set(BY_KIND) <= set(Features.names())
