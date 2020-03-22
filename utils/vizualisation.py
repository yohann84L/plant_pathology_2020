import random

import matplotlib.pyplot as plt

from datasets import PlantPathologyDataset
from datasets.transforms import UnNormalize

id_to_label = {
    0: "healthy",
    1: "multiple_diseases",
    2: "rust",
    3: "scab"
}


def plot_9_random_example(dataset: PlantPathologyDataset):
    n = 9
    idx = [random.randint(0, len(dataset)) for k in range(n)]

    cols, rows = 3, 3
    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(20, rows * 20 / 3))
    fig.tight_layout()
    for col in range(cols):
        for row in range(rows):
            img, label = dataset[idx[col + row]]
            id_label = label.argmax().item()
            label = id_to_label[id_label]
            img = UnNormalize()(img).numpy().transpose(1, 2, 0)
            ax[row, col].imshow(img)
            ax[row, col].axis("off")
            ax[row, col].set_title(f"Class : {label}")
    plt.show()
