def cer(ref: str, hyp: str) -> float:
    import editdistance
    return editdistance.eval(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    import editdistance
    return editdistance.eval(r, h) / max(1, len(r))
