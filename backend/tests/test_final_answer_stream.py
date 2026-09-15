"""Tests for streaming only the public field of a structured answer."""

from backend.app.services.final_answer_stream import InternalRefStreamFilter


def test_internal_ref_filter_hides_markers_split_across_chunks() -> None:
    stream = InternalRefStreamFilter()
    chunks = [
        "**First** [[plc_a1b2",
        "c3d4e5]]\n**Second** [[",
        "plc_f6e7d8c9b0]]",
    ]

    visible = [piece for chunk in chunks for piece in stream.feed(chunk)]
    visible.extend(stream.finish())

    assert "".join(visible) == "**First**\n**Second**"
    assert all("plc_" not in piece for piece in visible)


def test_internal_ref_filter_preserves_ordinary_markdown_brackets() -> None:
    stream = InternalRefStreamFilter()

    visible = stream.feed("See [") + stream.feed("details](https://example.com)")
    visible.extend(stream.finish())

    assert "".join(visible) == "See [details](https://example.com)"
