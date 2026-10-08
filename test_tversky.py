
import torch
import torch.nn.functional as F

from losses import TverskyLoss


def test_tversky():
    torch.manual_seed(42)

    loss_fn = TverskyLoss(
        idk=[1, 2, 3, 4],
        alpha=0.3,
        beta=0.7
    )

    labels = torch.randint(0, 5, (2, 16, 16))

    target = F.one_hot(
        labels, num_classes=5
    ).permute(0, 3, 1, 2).float()

    logits = torch.randn(2, 5, 16, 16, requires_grad=True)
    probs = torch.softmax(logits, dim=1)

    loss = loss_fn(probs, target)
    print("Tversky loss:", loss.item())

    assert torch.isfinite(loss)
    assert 0 <= loss.item() <= 1

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    print("Forward and backward passes: PASSED")


    perfect_loss = loss_fn(target.clone(), target)
    assert perfect_loss.item() < 1e-5
    print("Perfect prediction: PASSED")

    empty_target = torch.zeros_like(target)
    empty_target[:, 0] = 1.0

    empty_probs = torch.softmax(
        torch.randn(2, 5, 16, 16, requires_grad=True),
        dim=1
    )

    empty_loss = loss_fn(empty_probs, empty_target)
    assert torch.isfinite(empty_loss)
    assert empty_loss.item() == 0.0
    print("Empty foreground batch: PASSED")

    print("\nAll Tversky tests passed!")


if __name__ == "__main__":
    test_tversky()
