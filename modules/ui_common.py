import html


def plaintext_to_html(text, classname=None):
    content = "<br>\n".join(html.escape(x) for x in text.split('\n'))

    return f"<p class='{classname}'>{content}</p>" if classname else f"<p>{content}</p>"
