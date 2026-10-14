"""Temporary legacy forwarding entry point; study orchestration moved to REDbyRBD."""
def run_study(*args, **kwargs):
    try:
        from redbyrbd.trade_only_study import run_study as run
    except ImportError as error:
        raise ImportError("Study orchestration moved to REDbyRBD; install that research project.") from error
    return run(*args, **kwargs)
