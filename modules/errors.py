import sys
import textwrap
import threading
import traceback


exception_records = []
"""The 5 newest recorded exceptions (format_exception() dicts), oldest first; guarded by exception_records_lock."""
exception_records_lock = threading.Lock()


def format_traceback(tb):
    return [[f"{x.filename}, line {x.lineno}, {x.name}", x.line] for x in traceback.extract_tb(tb)]


def format_exception(e, tb):
    return {"exception": str(e), "traceback": format_traceback(tb)}


def get_exceptions():
    with exception_records_lock:
        return list(reversed(exception_records))


def record_exception(e: BaseException | None = None):
    """Records `e` (default: the exception being handled), unless it repeats the newest record: report() or display()
    followed by print_error_explanation() (or another report) for one exception records it once."""
    if e is None:
        e = sys.exception()
    if e is None:
        return

    record = format_exception(e, e.__traceback__)
    with exception_records_lock:
        if exception_records and exception_records[-1] == record:
            return

        exception_records.append(record)
        del exception_records[:-5]


def report(message: str, *, exc_info: bool = False) -> None:
    """
    Print an error message to stderr, with optional traceback.
    """

    record_exception()

    for line in message.splitlines():
        print("***", line, file=sys.stderr)
    if exc_info:
        print(textwrap.indent(traceback.format_exc(), "    "), file=sys.stderr)
        print("---", file=sys.stderr)


def print_error_explanation(message):
    record_exception()

    lines = message.strip().split("\n")
    max_len = max([len(x) for x in lines])

    print('=' * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print('=' * max_len, file=sys.stderr)


def display(e: Exception, task, *, full_traceback=False):
    # `e` is not necessarily the exception being handled (e.g. a cleanup failure displayed while handling another)
    record_exception(e)

    print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
    te = traceback.TracebackException.from_exception(e)
    if full_traceback:
        # include frames leading up to the try-catch block
        te.stack = traceback.StackSummary(traceback.extract_stack()[:-2] + te.stack)
    print(*te.format(), sep="", file=sys.stderr)

    message = str(e)
    if "copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])" in message:
        print_error_explanation("""
The most likely cause of this is you are trying to load Stable Diffusion 2.0 model without specifying its config file.
See https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20 for how to solve this.
        """)


def run(code, task):
    try:
        code()
    except Exception as e:
        display(e, task)
