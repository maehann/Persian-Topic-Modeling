from bertopic import BertTopic
import pandas as pd


df = pd.read_csv(
    r"D:\kaggle\isna\scraper\files\isna_complete.csv"
)

print(df.shape)

documents = (
    df["content"]
    .dropna()
    .astype(str)
    .tolist()[:10001:50]
)

print(f"Number of Documents: {len(documents)}")


model = BertTopic()


labels, topics = model.fit_transform(
    documents
)


print("\nTopic Labels:")
print(labels)

print(f"\nNumber of Topics: {len(topics)}")


print("\nTopic Words:")

for topic, words in topics.items():

    print(f"\nTopic {topic}")

    for word, score in words:

        print(f"{word:<25} {score:.4f}")


print("\nTopic 0:")
print(model.get_topic(0))