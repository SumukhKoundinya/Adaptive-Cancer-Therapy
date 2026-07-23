def tumor_burden(heaps):
    return sum(heaps)

def heterogeneity(heaps):
    return len(set(heaps)) / max(len(heaps), 1)

def clonal_entropy(heaps):
    from collections import Counter
    c = Counter(heaps)
    import math
    total = sum(c.values())
    return -sum((v/total) * math.log(v/total + 1e-9) for v in c.values())
