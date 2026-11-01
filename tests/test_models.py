import pytest
import torch
from drug_review.models import LSTMCell, ReviewClassifier, parameter_counts
from drug_review.data import rating_label, tokenize


@pytest.mark.parametrize("variant", ["vanilla", "cifg", "wmc"])
def test_equations_and_gradients(variant):
    torch.manual_seed(2)
    cell = LSTMCell(3, 4, variant).double()
    x = torch.randn(2, 3, dtype=torch.double)
    h = torch.randn(2, 4, dtype=torch.double)
    c = torch.randn(2, 4, dtype=torch.double)
    # Independent direct equation evaluation. Updated c is used in WMC output.
    z = x @ cell.input_linear.weight.T + cell.input_linear.bias + h @ cell.hidden_linear.weight.T
    if variant == "cifg":
        fz, oz, gz = z.chunk(3, -1)
        f = fz.sigmoid()
        expected_c = f*c + (1-f)*gz.tanh()
    else:
        iz, fz, oz, gz = z.chunk(4, -1)
        if variant == "wmc":
            iz = iz + (c @ cell.memory_i.weight.T).tanh()
            fz = fz + (c @ cell.memory_f.weight.T).tanh()
        expected_c = fz.sigmoid()*c + iz.sigmoid()*gz.tanh()
        if variant == "wmc":
            oz = oz + (expected_c @ cell.memory_o.weight.T).tanh()
    expected_h = oz.sigmoid()*expected_c.tanh()
    actual_h, actual_c = cell(x, (h,c))
    torch.testing.assert_close(actual_h, expected_h)
    torch.testing.assert_close(actual_c, expected_c)
    (actual_h.square().sum() + actual_c.square().sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in cell.parameters())


@pytest.mark.parametrize("variant", ["vanilla", "cifg", "wmc"])
def test_padding_invariance_and_script(variant):
    torch.manual_seed(42)
    model = ReviewClassifier(30, 5, 7, variant, dropout=0).eval()
    a = torch.tensor([[2,3,4]])
    b = torch.tensor([[2,3,4,0,0], [4,5,6,7,8]])
    torch.testing.assert_close(model(a,torch.tensor([3]))[0], model(b,torch.tensor([3,5]))[0])
    scripted = torch.jit.script(model)
    torch.testing.assert_close(scripted(b,torch.tensor([3,5])), model(b,torch.tensor([3,5])))


def test_parameter_formulas():
    for variant, gates in [("vanilla",4),("cifg",3),("wmc",4)]:
        model = ReviewClassifier(100,8,12,variant)
        expected = gates*12*(8+12+1) + (3*12*12 if variant=="wmc" else 0)
        assert parameter_counts(model)["recurrent"] == expected


@pytest.mark.parametrize("variant", ["vanilla", "cifg", "wmc"])
def test_sequence_optimization_preserves_gradients(variant):
    # Compare the optimized full-sequence projection/unbind against independent
    # per-word cell calls. This catches silent gradient changes in loop speedups.
    import copy
    torch.manual_seed(7)
    model=ReviewClassifier(20,4,5,variant,dropout=0.).double()
    reference=copy.deepcopy(model)
    tokens=torch.tensor([[2,3,4,0],[6,7,8,9]])
    lengths=torch.tensor([3,4])
    actual=model(tokens,lengths)
    h=torch.zeros(2,5,dtype=torch.double);c=torch.zeros_like(h)
    embedded=reference.embedding(tokens)
    for t in range(4):
        hn,cn=reference.cell(embedded[:,t],(h,c))
        active=(lengths>t).unsqueeze(1)
        h=torch.where(active,hn,h);c=torch.where(active,cn,c)
    expected=reference.classifier(h)
    torch.testing.assert_close(actual,expected)
    actual.square().sum().backward();expected.square().sum().backward()
    for (name,p),(ref_name,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==ref_name
        torch.testing.assert_close(p.grad,q.grad)


def test_labels_and_negation():
    assert [rating_label(i) for i in range(1,11)] == [0,0,0,0,1,1,2,2,2,2]
    assert "not" in tokenize("This is NOT helpful &amp; never worked.")
    assert "never" in tokenize("This is NOT helpful &amp; never worked.")
    with pytest.raises(ValueError):
        rating_label(4.5)
