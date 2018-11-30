import numpy as np

def dummy_pt_eta():
    np.random.seed(42)
    counts = np.random.exponential(2, size=50).astype(int)
    entries = np.sum(counts)
    test_in1 = np.random.uniform(-3., 3., size=entries)
    test_in2 = np.random.exponential(50., size=entries)+20.
    return (counts, test_in1, test_in2)

