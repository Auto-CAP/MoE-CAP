"""Filesystem paths shared by MoE-CAP recorder processes."""

import os
import tempfile


RECORDER_DIR_ENV = "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"


def get_vllm_recording_dir(environ=None):
    """Return the process-shared directory for vLLM recorder state.

    MoE-CAP launchers already assign a unique
    ``SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`` to every run.  Reuse that
    namespace for vLLM's multiprocessing control and data files so independent
    servers cannot start, stop, clear, or append to one another's recorder.

    Keep the historical system temporary-directory fallback for callers that
    do not set the environment variable.
    """

    env = os.environ if environ is None else environ
    configured_dir = env.get(RECORDER_DIR_ENV)
    if configured_dir:
        return os.path.abspath(os.path.expanduser(configured_dir))
    return tempfile.gettempdir()


def get_vllm_recording_path(filename, environ=None):
    """Return a recorder file path scoped to the current vLLM run."""

    return os.path.join(get_vllm_recording_dir(environ), filename)
