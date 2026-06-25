from datasets import load_dataset

def load_sst2():
    dataset = load_dataset("glue", "sst2")

    train_sentences = []
    train_labels = []

    for item in dataset["train"]:
        train_sentences.append(item["sentence"])
        train_labels.append(item["label"])

    return train_sentences, train_labels


if __name__ == "__main__":
    train_sentences, train_labels = load_sst2()

    print("Number of training examples:", len(train_sentences))
    print("\nFirst example:")
    print("Sentence:", train_sentences[0])
    print("Label:", train_labels[0])