"""WER/CER computation shared by both model types."""
import jiwer


def _normalize(refs: list[str], hyps: list[str]) -> tuple[list[str], list[str]]:
    # Replace empty refs to avoid jiwer division errors.
    refs = [r if r.strip() else " " for r in refs]
    return refs, hyps


def compute_wer_cer(refs: list[str], hyps: list[str]) -> dict:
    refs, hyps = _normalize(refs, hyps)
    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
    }
