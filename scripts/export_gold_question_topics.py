#!/usr/bin/env python3
import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results/hierarchical_topics/nn30_mcs20_seed2024"
DEFAULT_OUTPUT = PROJECT_DIR / "results/gold_questions/gold_questions_topic.csv"
DEFAULT_ORIGINAL_OUTPUT = PROJECT_DIR / "results/gold_questions/original_questions_topic.csv"


def read_csv(path):
    return pd.read_csv(path, dtype=str).fillna("")


def selected_name(row):
    option = row["selected_name_option"].strip().lower()
    if option in {"1", "2", "3"}:
        return row[f"suggestion_{option}"].strip()
    if option == "manual":
        return row["manual_name"].strip()
    return ""


def consolidated_topics(victoria_path, gabriel_path, adjudication_path):
    victoria = read_csv(victoria_path).set_index("topic_id", drop=False)
    gabriel = read_csv(gabriel_path).set_index("topic_id", drop=False)
    adjudication = read_csv(adjudication_path).set_index("topic_id", drop=False)
    if set(victoria.index) != set(gabriel.index):
        raise ValueError("Os arquivos de Victoria e Gabriel não contêm os mesmos tópicos.")
    if adjudication["relevance"].eq("").any():
        raise ValueError("A adjudicação do terceiro anotador ainda possui tópicos pendentes.")

    rows = []
    for topic_id in victoria.index:
        if topic_id in adjudication.index:
            final = adjudication.loc[topic_id]
        else:
            first, second = victoria.loc[topic_id], gabriel.loc[topic_id]
            if first["relevance"] != second["relevance"]:
                raise ValueError(f"Tópico {topic_id}: divergência de relevância sem adjudicação.")
            if first["relevance"] == "relevante" and selected_name(first) != selected_name(second):
                raise ValueError(f"Tópico {topic_id}: divergência de nome sem adjudicação.")
            final = first
        relevance = final["relevance"].strip().lower()
        name = selected_name(final) if relevance == "relevante" else ""
        if relevance not in {"relevante", "irrelevante"}:
            raise ValueError(f"Tópico {topic_id}: relevância inválida.")
        if relevance == "relevante" and not name:
            raise ValueError(f"Tópico {topic_id}: nome final ausente.")
        rows.append({"topic_id": int(topic_id), "topic_name": name, "relevance": relevance})
    return pd.DataFrame(rows).sort_values("topic_id").reset_index(drop=True)


def load_unique_questions(data_path):
    data = pd.read_csv(data_path)
    data = data[data["perguntas"].ne("[]")].copy()
    data["perguntas"] = data["perguntas"].apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)
    questions = data.explode("perguntas").dropna(subset=["perguntas"])["perguntas"].astype(str).reset_index(drop=True)
    codes, unique_questions = pd.factorize(questions, sort=False)
    first_indices = np.unique(codes, return_index=True)[1]
    return unique_questions.tolist(), first_indices


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Foram encontrados embeddings com norma zero.")
    return matrix / norms


def export(args):
    topics = consolidated_topics(args.victoria, args.gabriel, args.adjudication)
    relevant = topics.loc[topics["relevance"].eq("relevante")].copy()
    if relevant.empty:
        raise ValueError("Nenhum tópico foi mantido como relevante.")
    relevant["ordinal_topic_id"] = np.arange(1, len(relevant) + 1, dtype=int)
    ordinal_ids = relevant.set_index("topic_id")["ordinal_topic_id"]

    unique_questions, first_indices = load_unique_questions(args.questions)
    assignments = read_csv(args.assignments)
    if assignments["perguntas"].tolist() != unique_questions:
        raise ValueError("A ordem das perguntas únicas não corresponde ao arquivo de atribuições.")

    embeddings = np.load(args.embeddings, mmap_mode="r")
    if len(embeddings) <= int(first_indices.max()):
        raise ValueError("O arquivo de embeddings não corresponde ao corpus de perguntas.")
    unique_embeddings = normalize(np.asarray(embeddings[first_indices], dtype=np.float32))
    topic_ids = assignments["fine_topic_id"].astype(int).to_numpy()

    centroids = []
    central_examples = {}
    for topic_id in relevant["topic_id"]:
        member_indices = np.flatnonzero(topic_ids == topic_id)
        members = unique_embeddings[member_indices]
        if not len(members):
            raise ValueError(f"Tópico relevante {topic_id} não possui perguntas atribuídas.")
        centroid = normalize(members.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        closest = member_indices[np.argsort(members @ centroid)[::-1][:3]]
        if len(closest) < 3:
            raise ValueError(f"Tópico relevante {topic_id} possui menos de três perguntas.")
        central_examples[topic_id] = [unique_questions[index] for index in closest]
    centroids = normalize(np.vstack(centroids))

    gold_questions = [line.strip() for line in args.gold_questions.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_embeddings = normalize(np.asarray(np.load(args.gold_embeddings), dtype=np.float32))
    if len(gold_questions) != len(gold_embeddings):
        raise ValueError("A quantidade de perguntas padrão-ouro difere da quantidade de embeddings.")

    similarities = gold_embeddings @ centroids.T
    nearest = similarities.argmax(axis=1)
    relevant_ids = relevant["topic_id"].to_numpy(dtype=int)
    names = relevant.set_index("topic_id")["topic_name"]
    chosen_ids = relevant_ids[nearest]
    output = pd.DataFrame({
        "pergunta_padrao_ouro": gold_questions,
        "topic_id": [ordinal_ids.at[topic_id] for topic_id in chosen_ids],
        "topic_name": [names.at[topic_id] for topic_id in chosen_ids],
        "central_example_1": [central_examples[topic_id][0] for topic_id in chosen_ids],
        "central_example_2": [central_examples[topic_id][1] for topic_id in chosen_ids],
        "central_example_3": [central_examples[topic_id][2] for topic_id in chosen_ids],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    occurrences = read_csv(args.occurrences)
    occurrences["fine_topic_id"] = occurrences["fine_topic_id"].astype(int)
    occurrences = occurrences.loc[occurrences["fine_topic_id"].isin(relevant_ids)].copy()
    original_output = pd.DataFrame({
        "pergunta_original": occurrences["perguntas"],
        "topic_id": occurrences["fine_topic_id"].map(ordinal_ids).astype(int),
        "topic_name": occurrences["fine_topic_id"].map(names),
    })
    args.original_output.parent.mkdir(parents=True, exist_ok=True)
    original_output.to_csv(args.original_output, index=False)
    print(f"{len(output)} perguntas padrão-ouro salvas em {args.output}")
    print(f"{output['topic_id'].nunique()} tópicos relevantes receberam pelo menos uma pergunta")
    print(f"{len(original_output)} ocorrências de perguntas originais salvas em {args.original_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Associa perguntas padrão-ouro aos tópicos relevantes.")
    parser.add_argument("--questions", type=Path, default=PROJECT_DIR / "data/perguntas/relevant_question_extraction_gpt-5-4-mini_high_flex.csv")
    parser.add_argument("--gold-questions", type=Path, default=PROJECT_DIR / "data/files/queries.txt")
    parser.add_argument("--embeddings", type=Path, default=PROJECT_DIR / "results/hierarchical_topics/embeddings_all_questions_google_embeddinggemma-300m.npy")
    parser.add_argument("--gold-embeddings", type=Path, default=PROJECT_DIR / "results/hierarchical_topics/embeddings_gold_clustering.npy")
    parser.add_argument("--assignments", type=Path, default=RESULTS_DIR / "unique_questions_with_fine_topics.csv")
    parser.add_argument("--occurrences", type=Path, default=RESULTS_DIR / "all_question_occurrences_with_topics.csv")
    parser.add_argument("--victoria", type=Path, default=PROJECT_DIR / "results/manual_annotation/annotators/topic_annotation_victoria.csv")
    parser.add_argument("--gabriel", type=Path, default=PROJECT_DIR / "results/manual_annotation/annotators/topic_annotation_Gabriel.csv")
    parser.add_argument("--adjudication", type=Path, default=PROJECT_DIR / "results/annotation_agreement/annotators/topic_annotation_terceiro_anotador.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--original-output", type=Path, default=DEFAULT_ORIGINAL_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
