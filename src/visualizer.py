import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURE_SIZE,
    CSV_ENCODING
)

class TopicVisualizer:
    def __init__(
        self,
        topic_model,
        statistics=None,
        evaluation=None
    ):

        self.topic_model = topic_model
        self.statistics = statistics
        self.evaluation = evaluation
        self.figures = {}

    # ------------------------
    # Private Methods
    # ------------------------

    def _apply_plot_style(self):

        plt.figure(figsize=FIGURE_SIZE)
        plt.grid(
            axis="y",
            linestyle="--",
            alpha=0.3
        )


    def _save_figure(self, path):

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=FIGURE_DPI,
            format=FIGURE_FORMAT,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved successfully:\n{path}")

    # ------------------------
    # Dataset Visualizations
    # ------------------------



    # ------------------------
    # Custom Visualizations
    # ------------------------

    def plot_topic_sizes(self):

        table = self.statistics.topic_sizes()
        self._apply_plot_style()

        plt.bar(
            table["Topic"].astype(str),
            table["Count"]
        )

        plt.xlabel("Topic")
        plt.ylabel("Number of Documents")
        plt.title("Topic Sizes")
        self.figures["topic_sizes"] = plt.gcf()

        return plt.gcf()
    

    def plot_topic_percentages(self):

        table = self.statistics.topic_percentages()
        self._apply_plot_style()

        plt.bar(
            table["Topic"].astype(str),
            table["Percentage"]
        )

        plt.xlabel("Topic")
        plt.ylabel("Percentage (%)")
        plt.title("Topic Distribution")
        self.figures["topic_percentages"] = plt.gcf()

        return plt.gcf()
    

    def plot_category_distribution(self):

        table = self.statistics.topic_category_distribution()

        pivot = table.pivot(
            index="Topic",
            columns="category",
            values="Percentage"
        ).fillna(0)

        self._apply_plot_style()

        pivot.plot(
            kind="bar",
            stacked=True
        )

        plt.xlabel("Topic")
        plt.ylabel("Percentage (%)")
        plt.title("Category Distribution per Topic")

        self.figures["category_distribution"] = plt.gcf()

        return plt.gcf()
    

    def plot_subcategory_distribution(self):

        table = self.statistics.topic_subcategory_distribution()

        pivot = table.pivot(
            index="Topic",
            columns="sub_category",
            values="Percentage"
        ).fillna(0)

        self._apply_plot_style()

        pivot.plot(
            kind="bar",
            stacked=True
        )

        plt.xlabel("Topic")
        plt.ylabel("Percentage (%)")
        plt.title("Subcategory Distribution per Topic")
        self.figures["subcategory_distribution"] = plt.gcf()

        return plt.gcf()
    

    def plot_evaluation_metrics(self):

        table = self.evaluation.metrics_df.copy()
        self._apply_plot_style()

        plt.bar(
            table["Metric"],
            tabl
            e["Value"]
        )
        plt.xticks(rotation=30)
        plt.ylabel("Score")
        plt.title("Evaluation Metrics")
        self.figures["evaluation_metrics"] = plt.gcf()

        return plt.gcf()
