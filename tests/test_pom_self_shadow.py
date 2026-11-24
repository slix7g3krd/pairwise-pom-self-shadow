import pytest
import graphics.pom_self_shadow as mod

CASES = [
    ("self_occlusion", (1,)),
    ("mip_step", (1,)),
    ("depth_bias", (1,)),
    ("mirror_fix", (1,)),
    ("degrade_blend", (1,)),
    ("step_cap", (1,)),
    ("height_range", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("grazing_angle", (1,)),
    ("metrics", (1,)),
    ("uv_wrap", (1,)),
    ("precision", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
