from sentence_transformers import SentenceTransformer


def main() -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "The weather is lovely today.",
        "It's so sunny outside!",
        "He drove to the stadium.",
    ]

    embeddings = model.encode(sentences)
    print("Embeddings shape:", embeddings.shape)

    similarities = model.similarity(embeddings, embeddings)
    print("Similarity matrix:")
    print(similarities)


if __name__ == "__main__":
    main()
