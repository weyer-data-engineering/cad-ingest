"""Finding the sheet of a scan on a scanner lid about as grey as its paper (sheet.on_lid)."""
import cv2
import numpy as np
import pytest
from synth import plot, printed_on, scan

from app.plan2cad import sheet
from app.plan2cad.model import PAPER

DPI = 100


def frame_at(gray: np.ndarray, found: sheet.Sheet, ppm: float) -> tuple[float, float]:
    """Where the straightened sheet has the top and the left line of the drawing frame (mm): the darkest row and
    column near them."""
    flat = sheet.rectify(gray, found, ppm).astype(np.float32)
    rows = flat[round(5 * ppm):round(15 * ppm), round(100 * ppm):round(300 * ppm)].mean(axis=1)
    cols = flat[round(50 * ppm):round(200 * ppm), round(15 * ppm):round(25 * ppm)].mean(axis=0)
    return (np.argmin(rows) + round(5 * ppm)) / ppm, (np.argmin(cols) + round(15 * ppm)) / ppm


@pytest.mark.parametrize("lid, skew", [(210, 0.8), (210, 4.0), (235, 0.8)])
def test_paper_on_a_lid_as_grey_as_it_is_found_by_its_straight_edges(plan, lid, skew):
    img = scan(plan, 7, dpi=DPI, poor=True, skew=skew, lid=lid)  # the paper of a poor scan is 222
    assert sheet.locate(img, PAPER["A3"]) is None or sheet.locate(img, PAPER["A3"]).method == "frame"  # as it was
    found = sheet.locate(img, PAPER["A3"], lid=True)
    assert found.method == "paper" and found.size == (420, 297)
    assert sheet.source_ppm(found) == pytest.approx(DPI / 25.4, rel=0.003)
    top, left = frame_at(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), found, DPI / 25.4)
    assert top == pytest.approx(10, abs=0.6) and left == pytest.approx(20, abs=0.6)  # where the frame was drawn


def test_a_scan_cropped_to_its_sheet_has_no_lid_even_with_a_broad_border(plan):
    img = plot(plan, DPI)
    cv2.rectangle(img, (round(4 * DPI / 25.4),) * 2, (img.shape[1] - round(4 * DPI / 25.4),
                                                       img.shape[0] - round(4 * DPI / 25.4)), (0, 0, 0), 12)
    assert sheet.locate(img, PAPER["A3"], lid=True) is None


def test_a_dark_lid_is_still_told_apart_by_its_colour(plan):
    found = sheet.locate(scan(plan, 7, dpi=DPI, poor=True), PAPER["A3"], lid=True)  # a lid of 150
    assert found.method == "paper" and sheet.source_ppm(found) == pytest.approx(DPI / 25.4, rel=0.003)


def on_bed(page: np.ndarray, dx: float, dy: float, lid: float = 210) -> np.ndarray:
    """A poor copy (paper 222) of ``page`` on an A3 flatbed (landscape) with a grey lid, ``dx``, ``dy`` mm from its
    top left corner."""
    ppm = DPI / 25.4
    img = np.full((round(297 * ppm), round(420 * ppm)), lid, np.float32)
    x, y = round(dx * ppm), round(dy * ppm)
    part = page[:img.shape[0] - y, :img.shape[1] - x].astype(np.float32)
    img[y:y + part.shape[0], x:x + part.shape[1]] = 95 + (222 - 95) * part / 255
    img = cv2.GaussianBlur(img, (0, 0), 1.1) + np.random.default_rng(5).normal(0, 7, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize("dx, dy", [(2, 3), (0, 4), (3, 0)])
def test_a_small_sheet_near_the_corner_of_a_large_bed_is_not_the_bed(plan, dx, dy):
    page = cv2.cvtColor(plot(printed_on(plan, "A4"), DPI), cv2.COLOR_BGR2GRAY)
    img = on_bed(np.ascontiguousarray(np.rot90(page)), dx, dy)  # A4 portrait: the lid shows on its right
    assert sheet.locate(img, PAPER["A4"], lid=True) is None  # as before: the dpi gives the scale


@pytest.mark.parametrize("sides, grey, mm", [("left", 60, 5), ("left", 120, 4), ("bottom", 90, 3)])
def test_a_copy_with_a_dark_band_along_an_edge_is_taken_as_it_is(plan, sides, grey, mm):
    img = plot(plan, DPI)
    b = round(mm * DPI / 25.4)
    img[(np.s_[:, :b] if sides == "left" else np.s_[-b:, :])] = grey
    assert sheet.locate(img, PAPER["A3"], lid=True) is None


def test_paper_that_runs_out_of_the_image_is_not_stretched_to_its_size(plan):
    img = scan(plan, 7, dpi=DPI, poor=True, lid=210)
    whole = sheet.locate(img, PAPER["A3"], lid=True)
    right = round(max(whole.corners[1][0], whole.corners[2][0]) - 5 * DPI / 25.4)
    assert sheet.locate(np.ascontiguousarray(img[:, :right]), PAPER["A3"], lid=True) is None


def test_paper_on_a_lid_must_agree_with_a_known_dpi(plan):
    img = scan(plan, 7, dpi=DPI, poor=True, lid=210)
    assert sheet.locate(img, PAPER["A3"], lid=True, dpi=DPI).method == "paper"
    assert sheet.locate(img, PAPER["A3"], lid=True, dpi=1.03 * DPI) is None  # the dpi is taken instead
