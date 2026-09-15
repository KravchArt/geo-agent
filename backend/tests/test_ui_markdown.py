"""UI Markdown display normalization tests."""

from ui.markdown import prepare_chat_markdown


def test_organisation_branches_use_visible_markdown_line_breaks() -> None:
    text = "**The Бык**\n• Комсомольский проспект\n• Ветошный переулок"

    assert prepare_chat_markdown(text) == (
        "**The Бык**  \n• Комсомольский проспект  \n• Ветошный переулок"
    )


def test_existing_paragraph_breaks_and_ordinary_lines_are_preserved() -> None:
    text = "Вступление\n\n**The Бык**\n• Арбат\n\nИтог"

    assert prepare_chat_markdown(text) == "Вступление\n\n**The Бык**  \n• Арбат\n\nИтог"
