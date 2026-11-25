"""验证业务输入之间的引用关系及形状，不计算目标输出。"""
import numpy as np
from support.inspect_data import load_bundle

def test_linked_materials():
    _, arrays, r, scenarios = load_bundle()
    assert np.asarray(r['valid']).shape == (arrays['features'].shape[0],*arrays['features'].shape[2:])
    cameras={f['camera_id'] for f in r['frames']}
    assert all(cameras<=set(m['mapping']) for m in r['domain_maps'])
    assert len({f['frame_id'] for f in r['frames']}) == arrays['features'].shape[0]
    assert arrays['conv_input'].shape == arrays['features'].shape
