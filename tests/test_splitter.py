from ekgclf.data.splitter import patient_level_split

def test_patient_level_split_determinism():
    pids = [1,2,3,4,5,6,7,8,9,10]
    a = patient_level_split(pids, 0.8, 0.1, 0.1, seed=123)
    b = patient_level_split(pids, 0.8, 0.1, 0.1, seed=123)
    assert a.train == b.train and a.val == b.val and a.test == b.test
