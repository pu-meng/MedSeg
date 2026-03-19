import torch

from medseg.models.build_model import build_model


def run_one_model(name, out_channels=2):
    print(f"\n[TEST] model={name}")
    model = build_model(
        name=name,
        in_channels=1,
        out_channels=out_channels,
        img_size=(96, 96, 96),
    )
    model.eval()

    x = torch.randn(1, 1, 96, 96, 96)

    with torch.no_grad():
        y = model(x)

    print("output shape:", tuple(y.shape))
    assert y.shape[0] == 1
    assert y.shape[1] == out_channels
    assert y.shape[2:] == (96, 96, 96)


def main():
    test_models = [
        "unet3d",
        "attention_unet",
        "segresnet",
        "dynunet",
    ]

    for name in test_models:
        run_one_model(name)

    # transformer 类模型更吃内存，单独测
    run_one_model("unetr")
    run_one_model("swinunetr")

    print("\nAll forward tests passed.")


if __name__ == "__main__":
    main()
