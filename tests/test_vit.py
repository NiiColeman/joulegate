import torch

from models.lora import LoRALinear, add_lora
from models.vit_tiny import VIT_TARGETS, TinyViT


def test_tinyvit_runs_at_96_pixels_and_splits_into_prefix_and_suffix():
    torch.manual_seed(0)
    m = TinyViT(pretrained=False).eval()
    x = torch.randn(3, 3, 96, 96)
    with torch.no_grad():
        full = m(x)
        h = m.run_blocks(m.embed(x), 0, 10)
        split = m.head_out(m.run_blocks(h, 10))
    assert full.shape == (3, 10) and h.shape == (3, 37, 192)
    torch.testing.assert_close(full, split)


def test_lora_on_last_two_vit_blocks_plus_head():
    m = TinyViT(pretrained=False)
    params, first = add_lora(m, 2, targets=VIT_TARGETS, train_head=True)
    assert first == len(m.blocks) - 2
    wrapped = [n for n, mod in m.named_modules() if isinstance(mod, LoRALinear)]
    assert len(wrapped) == 2 * len(VIT_TARGETS)
    assert all(n.startswith(("net.blocks.10", "net.blocks.11")) for n in wrapped)
    head = {id(p) for p in m.head.parameters()}
    assert head <= {id(p) for p in params}
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in params}


def test_head_only_lora_does_not_duplicate_head_parameters():
    m = TinyViT(pretrained=False)
    params, _ = add_lora(m, 0, targets=VIT_TARGETS, train_head=True)
    assert len(params) == len({id(p) for p in params}) == len(list(m.head.parameters()))
