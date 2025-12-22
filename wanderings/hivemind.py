"""
Artificial Hivemind - Explore multi-agent conversation datasets
Uses data from: https://huggingface.co/collections/liweijiang/artificial-hivemind
"""

from datasets import load_dataset


def explore_dataset(dataset_name):
    """Explore the structure of the dataset"""
    print("=" * 60)
    print(f"\n\n\n\n\n\n\nDataset Structure {dataset_name}")
    print("=" * 60)

    dataset = load_dataset(dataset_name, split="train")

    # Show first example
    for i in range(3):
        example = dataset[0]
        print(f"\nFirst example keys: {list(example.keys())}\n")

        for key, value in example.items():
            print(f"{key}:")
            if isinstance(value, (list, dict)):
                print(f"  Type: {type(value).__name__}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"  Length: {len(value)}")
                    print(f"  First item: {value[0]}")
            else:
                print(f"  {value}")
            print()


def main():
    explore_dataset("liweijiang/infinite-chats-taxonomy")
    explore_dataset("liweijiang/infinite-chats-eval")
    explore_dataset("liweijiang/infinite-chats-human-absolute")
    explore_dataset("liweijiang/infinite-chats-human-pairwise")


if __name__ == "__main__":
    main()
