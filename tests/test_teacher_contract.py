"""External-teacher input/label contract fixtures, not accuracy evidence."""
import numpy as np
from scripts.screen_expression_teacher import fer_order,teacher_blob

def test_external_native_labels_map_to_fer_without_neutral_sad_swap():
    scores=np.eye(7,dtype=np.float32)
    assert fer_order(scores).argmax(1).tolist()==[0,1,2,3,6,4,5]

def test_teacher_input_black_white_range_and_rgb_replication():
    blob=teacher_blob([np.zeros((48,48),dtype=np.uint8),np.full((48,48),255,dtype=np.uint8)])
    assert blob.shape==(2,3,112,112)
    assert np.allclose(blob[0],-1.) and np.allclose(blob[1],1.)
    assert np.array_equal(blob[:,0],blob[:,1]) and np.array_equal(blob[:,1],blob[:,2])
