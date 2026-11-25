"""核对合成输入的业务引用及文件一致性。"""
import numpy as np
from support.inspect_data import load_bundle

def test_context():
    _, arrays, r, scenarios = load_bundle()
    parameters={v['parameter_id']:v for v in r['parameters']}
    members={v['member_id'] for v in r['members']}
    for block in r['blocks']:
        assert block['member_id'] in members
        assert arrays[block['tensor']].shape == tuple(parameters[block['parameter_id']]['shape'])
    assert all(v['sample_count']>=0 for v in r['members'])
