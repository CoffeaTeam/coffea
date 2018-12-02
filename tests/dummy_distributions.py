import numpy as np

def dummy_pt_eta():
    np.random.seed(42)
    counts = np.random.exponential(2, size=50).astype(int)
    entries = np.sum(counts)
    test_in1 = np.random.uniform(-3., 3., size=entries)
    test_in2 = np.random.exponential(10., size=entries)+np.random.exponential(10, size=entries)
    return (counts, test_in1, test_in2)

def dummy_four_momenta():
    np.random.seed(12345)
    nrows = 1000
    counts = np.minimum(np.random.exponential(0.5, size=nrows).astype(int), 20)
    
    px = np.random.normal(loc=20.0,scale=5.0,size=np.sum(counts))
    py = np.random.normal(loc=20.0,scale=5.0,size=np.sum(counts))
    pz = np.random.normal(loc=0, scale=55, size=np.sum(counts))
    m_pi = np.full_like(px,fill_value=0.135)
    energy = np.sqrt(px*px + py*py + pz*pz + m_pi*m_pi)
    return (counts,px,py,pz,energy)

def dummy_events():
    counts, px, py, pz, energy = dummy_four_momenta()
    thep4 = np.stack((px,py,pz,energy)).T

    class obj(object):
        def __init__(self):
            self.p4 = thep4
            self.blah = energy*px
            self.count = counts
    
    class events(object):
        def __init__(self):
            self.thing = obj()
    

    return events()
