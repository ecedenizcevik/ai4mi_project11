
import torch
import torch.nn.functional as F
from losses import FocalLoss, CrossEntropy

def test_focal():
    torch.manual_seed(42)

    labels = torch.randint(0, 5, (2, 16, 16))
    target = F.one_hot(labels, num_classes=5)
    target = target.permute(0, 3, 1, 2).float()
    logits = torch.randn(2, 5, 16, 16, requires_grad=True)
    probs = F.softmax(logits, dim=1)
    loss_fn = FocalLoss(idk=list(range(5)), gamma=2.0)
    loss = loss_fn(probs, target)

    print("Focal loss:", loss.item())
    assert torch.isfinite(loss)
    assert loss.item() >= 0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    print("Forward and backward: PASSED")

    focal_ce = FocalLoss(idk=list(range(5)), gamma=0.0)
    ce = CrossEntropy(idk=list(range(5)))
    assert torch.allclose(
        focal_ce(probs.detach(), target),
        ce(probs.detach(), target),
        atol=1e-6
    )
    print("Gamma=0 equals CE: PASSED")
    perfect_loss = loss_fn(target.clone(), target)
    assert perfect_loss.item() < 1e-5
    print("Perfect prediction: PASSED")
    print("\nAll Focal Loss tests passed!")


if __name__ == "__main__":
    test_focal()
