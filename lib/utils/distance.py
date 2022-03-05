from polyleven import levenshtein


def is_similar(seq_a, seq_b, dist_type="edit", threshold=0):
    if dist_type == "edit":
        return edit_dist(seq_a, seq_b) < threshold
    return False


def edit_dist(seq1, seq2):
    return levenshtein(seq1, seq2) / 1
