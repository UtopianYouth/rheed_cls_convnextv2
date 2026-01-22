import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default = 1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    tfm = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    dataset = FakeData(
        size=256,
        image_size=(3, args.img_size, args.img_size),
        num_classes=args.num_classes,
        transform=tfm
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    import timm
    model = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        running = 0.0
        for step, (x, y) in enumerate(loader, start=1):
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()

            running += loss.item()
            if step % 20 == 0:
                print(f"epoch {epoch+1} step {step} loss {running/20:.4f}")
                running = 0.0

        print(f"epoch {epoch+1} done, time {time.time()-t0:.1f}s")

    print("Sanity check finished OK.")

if __name__ == "__main__":
    main()
