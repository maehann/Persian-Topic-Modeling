import numpy as np
import pandas as pd
import os
from config import (
    CATEGORY_COLUMN,
    SUBCATEGORY_COLUMN,
    TEXT_COLUMN,
    SUMMARY_COLUMN,
    CSV_ENCODING
)

class TopicStatistics:
    def __init__(
    self,
    topic_model,
    documents,
    topics,
    probabilities=None,
    dataframe=None):
        
        self.topic_model = topic_model
        self.documents = documents
        self.topics = np.array(topics)
        self.probabilities = probabilities
        self.df = dataframe
        # Cache
        self.topic_info = self.topic_model.get_topic_info()
        # Store generated tables
        self.tables = {}
        self.vocabulary_size = len(
        {
            word
            for doc in self.documents
            for word in doc.split()
        }
)


    # ------------------------
    # Private Methods
    # ------------------------

    def _get_topic_words(self):

        topic_dict = self.topic_model.get_topics()
        topic_words = {}

        for topic_id, words in topic_dict.items():

            if topic_id == -1:
                continue

            topic_words[topic_id] = [
                word for word, _ in words
            ]

        return topic_words
    

    def _save_dataframe(self, dataframe, path):

        dataframe.to_csv(
            path,
            index=False,
            encoding=CSV_ENCODING
        )

        print( f"Saved successfully:\n{path}")


    # ------------------------
    # Dataset Statistics
    # ------------------------

    def dataset_statistics(self):

        statistics = {

            "Number of Documents":
                len(self.documents),

            "Number of Categories":
                self.df[CATEGORY_COLUMN].nunique(),

            "Number of Subcategories":
                self.df[SUBCATEGORY_COLUMN].nunique(),

            "Number of Topics":
                len(self.topic_info[self.topic_info["Topic"] != -1]),

            "Outlier Documents":
                np.sum(self.topics == -1),

            "Outlier Percentage (%)":
                round(
                    (np.sum(self.topics == -1) / len(self.documents)) * 100,
                    2
                ),

            "Average Document Length":
                round(
                    np.mean(
                        [len(doc.split()) for doc in self.documents]
                    ),
                    2
                ),

            "Vocabulary Size": 
                self.vocabulary_size
        }

        table = pd.DataFrame(
            statistics.items(),
            columns=[
                "Statistic",
                "Value"
            ]
        )

        self.tables["dataset_statistics"] = table

        return table

    # ------------------------
    # Topic Statistics
    # ------------------------

    def topic_information(self):

        table = self.topic_info.copy()
        self.tables["topic_information"] = table

        return table
    

    def topic_keywords(self, top_n=10):

        rows = []
        topic_words = self._get_topic_words()
        for topic_id, words in topic_words.items():

            rows.append({
                "Topic": topic_id,
                "Keywords": ", ".join(words[:top_n])
            })

        table = pd.DataFrame(rows)
        self.tables["topic_keywords"] = table

        return table
    

    def topic_sizes(self):

        table = (
            self.topic_info[
                self.topic_info["Topic"] != -1
            ][["Topic", "Count"]]
            .copy()
            .sort_values("Count", ascending=False)
            .reset_index(drop=True)
        )
        self.tables["topic_sizes"] = table

        return table
    

    def topic_percentages(self):

        table = self.topic_sizes().copy()
        table["Percentage"] = (table["Count"] / len(self.documents)*100).round(2)
        self.tables["topic_percentages"] = table

        return table
    

    def representative_documents(self, top_n=3):

        document_info = self.topic_model.get_document_info(
            self.documents
        )

        document_info = (
            document_info[document_info["Topic"] != -1]
            .sort_values(
                ["Topic", "Probability"],
                ascending=[True, False]
            )
            .groupby("Topic")
            .head(top_n)
            .reset_index(drop=True)
        )

        text_column = (
            "Document"
            if "Document" in document_info.columns
            else TEXT_COLUMN
        )
        
        table = document_info[
            [
                "Topic",
                "Probability",
                text_column
            ]
        ].copy()

        self.tables["representative_documents"] = table

        return table
    

    def average_document_length_per_topic(self):

        df = pd.DataFrame({

            "Topic": self.topics,

            "Length": [
                len(doc.split())
                for doc in self.documents
            ]
        })

        table = (
            df[df["Topic"] != -1]
            .groupby("Topic")
            .agg(
                Average_Length=("Length", "mean"),
                Number_of_Documents=("Length", "count")
            )
            .reset_index()
        )

        table["Average_Length"] = (
            table["Average_Length"]
            .round(2)
        )

        self.tables["average_document_length_per_topic"] = table

        return table
    

    def topic_category_distribution(self):

        table = pd.DataFrame({
            "Topic": self.topics,
            CATEGORY_COLUMN:
                self.df[CATEGORY_COLUMN]

        })

        table = (
            table[table["Topic"] != -1]
            .groupby(
                ["Topic", CATEGORY_COLUMN]
            )
            .size()
            .reset_index(name="Count")
        )

        table["Percentage"] = (
            table.groupby("Topic")["Count"]
            .transform(
                lambda x: round(
                    x / x.sum() * 100,
                    2
                )
            )
        )

        self.tables["topic_category_distribution"] = table

        return table
    

    def topic_subcategory_distribution(self):

        table = pd.DataFrame({

            "Topic": self.topics,

            SUBCATEGORY_COLUMN:
                self.df[SUBCATEGORY_COLUMN]

        })

        table = (
            table[table["Topic"] != -1]
            .groupby(
                ["Topic", SUBCATEGORY_COLUMN]
            )
            .size()
            .reset_index(name="Count")
        )

        table["Percentage"] = (
            table.groupby("Topic")["Count"]
            .transform(
                lambda x: round(
                    x / x.sum() * 100,
                    2
                )
            )
        )

        self.tables["topic_subcategory_distribution"] = table

        return table

    # ------------------------
    # Utility
    # ------------------------

    def generate_all_tables(self):

        self.dataset_statistics()
        self.topic_information()
        self.topic_keywords()
        self.topic_sizes()
        self.topic_percentages()
        self.representative_documents()
        self.average_document_length_per_topic()
        self.topic_category_distribution()
        self.topic_subcategory_distribution()

        return self.tables
    

    def save_table(self, table_name, path):

        if table_name not in self.tables:

            raise ValueError(
                f"Table '{table_name}' does not exist. "
                "Generate it first."
            )

        self._save_dataframe(
            self.tables[table_name],
            path
        )


    def save_all_tables(self, output_directory):

        os.makedirs(
            output_directory,
            exist_ok=True
        )

        if not self.tables:
            self.generate_all_tables()

        for table_name, dataframe in self.tables.items():

            file_path = os.path.join(
                output_directory,
                f"{table_name}.csv"
            )
            self._save_dataframe(
                dataframe,
                file_path
            )

        print(
            f"\nAll tables were saved to:\n{output_directory}"
        )