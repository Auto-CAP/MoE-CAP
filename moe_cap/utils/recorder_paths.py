"""Filesystem paths shared by MoE-CAP recorder processes."""

import os
import tempfile


RECORDER_DIR_ENV = "SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"

SGLANG_RECORD_FILENAME = "expert_distribution_record.jsonl"
_SGLANG_DEFAULT_SUBDIR = "expert_records"


def get_sglang_recorder_dir(environ=None):
    """Return the directory the SGLang server writes forward-pass records to.

    Server and client are separate processes that rendezvous on this directory,
    so both must resolve it identically: the server writes the record file here
    and the client reads it back to obtain the authoritative TTFT, per-pass
    latency, batch sizes and expert activation.  Resolving it in two places let
    the two defaults drift apart, and the client then read an absent file,
    silently publishing zeroed batch sizes and null timings.

    Under ``--profiling-only`` this is the handoff for dense timing records too,
    not only expert distribution, so it applies to every SGLang run.
    """

    env = os.environ if environ is None else environ
    configured_dir = env.get(RECORDER_DIR_ENV)
    if configured_dir:
        return os.path.abspath(os.path.expanduser(configured_dir))
    return os.path.abspath(
        os.path.join(os.path.expanduser("~"), _SGLANG_DEFAULT_SUBDIR)
    )


def get_sglang_record_path(model_path, environ=None):
    """Return the SGLang forward-record file for ``model_path``."""

    return os.path.join(
        get_sglang_recorder_dir(environ), model_path, SGLANG_RECORD_FILENAME
    )


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
