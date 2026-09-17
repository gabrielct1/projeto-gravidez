#!/usr/bin/env python3
import argparse
import fcntl
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CSV = PROJECT_DIR / "results/manual_annotation/topic_annotation.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "results/manual_annotation/annotators"
ANNOTATION_FIELDS = ["relevance", "selected_name_option", "manual_name", "notes", "annotator_id", "annotated_at"]
EXAMPLE_FIELDS = [f"central_example_{i}" for i in range(1, 6)] + [f"boundary_example_{i}" for i in range(1, 6)]
ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def read_csv(path):
    return pd.read_csv(path, dtype=str).fillna("")


def validate_annotator_id(annotator_id):
    if not ID_PATTERN.fullmatch(annotator_id):
        raise ValueError("O ID deve conter apenas letras, números, _, - ou . e ter até 64 caracteres.")


def validate_structure(data, base):
    if list(data.columns) != list(base.columns):
        raise ValueError("As colunas do CSV do anotador não correspondem ao formulário-base.")
    if len(data) != len(base) or not data["topic_id"].equals(base["topic_id"]):
        raise ValueError("Os tópicos do CSV do anotador não correspondem ao formulário-base.")
    static_fields = [column for column in base.columns if column not in ANNOTATION_FIELDS]
    if not data[static_fields].equals(base[static_fields]):
        raise ValueError("Exemplos ou sugestões do CSV do anotador diferem do formulário-base.")


def validate_completed(data):
    errors = []
    for row in data.itertuples(index=False):
        relevance = row.relevance.strip().lower()
        option = row.selected_name_option.strip().lower()
        if relevance not in {"relevante", "irrelevante"}:
            errors.append(f"Tópico {row.topic_id}: relevância ausente ou inválida.")
        elif relevance == "relevante":
            if option not in {"1", "2", "3", "manual"}:
                errors.append(f"Tópico {row.topic_id}: opção de nome ausente ou inválida.")
            elif option == "manual" and not row.manual_name.strip():
                errors.append(f"Tópico {row.topic_id}: nome manual ausente.")
        examples = [getattr(row, field) for field in EXAMPLE_FIELDS]
        if any(not value.strip() for value in examples) or len(examples) != len(set(examples)):
            errors.append(f"Tópico {row.topic_id}: exemplos ausentes ou repetidos.")
    return errors


def atomic_save(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", suffix=".csv", dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            data.to_csv(file, index=False)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


@contextmanager
def annotator_lock(output_dir, annotator_id):
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f"topic_annotation_{annotator_id}.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Já existe uma sessão ativa para o anotador {annotator_id}.") from error
        lock.write(str(os.getpid()))
        lock.flush()
        yield


def load_or_create(base_path, personal_path, annotator_id):
    if not base_path.exists():
        raise FileNotFoundError(f"Formulário-base não encontrado: {base_path}")
    base = read_csv(base_path)
    if personal_path.exists():
        data = read_csv(personal_path)
        validate_structure(data, base)
    else:
        data = base.copy()
        for field in ANNOTATION_FIELDS:
            data[field] = ""
        atomic_save(data, personal_path)
    foreign_ids = set(data.loc[data["annotator_id"].ne(""), "annotator_id"]) - {annotator_id}
    if foreign_ids:
        raise ValueError(f"O arquivo contém respostas de outro anotador: {sorted(foreign_ids)}")
    return data


def last_annotation(data):
    completed = data.loc[data["relevance"].ne("")].copy()
    if completed.empty:
        return None
    completed["parsed_time"] = pd.to_datetime(completed["annotated_at"], errors="coerce", utc=True)
    if completed["parsed_time"].notna().any():
        return completed.loc[completed["parsed_time"].idxmax()]
    return completed.iloc[-1]


def next_pending(data, current=-1, skipped=None):
    skipped = skipped or set()
    size = len(data)
    for offset in range(1, size + 1):
        index = (current + offset) % size
        if not data.at[index, "relevance"].strip() and index not in skipped:
            return index
    return None


def show_cluster(row, completed, total, output, adjudication=False):
    output("\n" + "=" * 88)
    output(f"Tópico {row.topic_id} | tamanho {row.topic_size} | concluídos {completed}/{total}")
    output(f"Palavras principais: {row.top_words}")
    output("\nPerguntas centrais")
    for i in range(1, 6):
        output(f"  C{i}. {getattr(row, f'central_example_{i}')}")
    output("\nPerguntas de borda")
    for i in range(1, 6):
        output(f"  B{i}. {getattr(row, f'boundary_example_{i}')}")
    if not adjudication:
        output("\nSugestões de nome")
        for i in range(1, 4):
            output(f"  {i}. {getattr(row, f'suggestion_{i}')}")
    if row.relevance:
        chosen = row.manual_name if row.selected_name_option == "manual" else getattr(row, f"suggestion_{row.selected_name_option}", "")
        output(f"\nAnotação atual: {row.relevance}" + (f" | nome: {chosen}" if chosen else ""))
    if row.notes:
        output(f"Observação: {row.notes}")


def edit_note(data, index, annotator_id, personal_path, input_fn, output):
    current = data.at[index, "notes"]
    if current:
        output(f"Observação atual: {current}")
    data.at[index, "notes"] = input_fn("Nova observação (vazio remove): ").strip()
    data.at[index, "annotator_id"] = annotator_id
    atomic_save(data, personal_path)


def annotate_relevant(data, index, annotator_id, personal_path, input_fn, output):
    while True:
        choice = input_fn("Nome [1/2/3, n=novo, o=observação, b=voltar, q=sair]: ").strip().lower()
        if choice in {"1", "2", "3"}:
            data.at[index, "relevance"] = "relevante"
            data.at[index, "selected_name_option"] = choice
            data.at[index, "manual_name"] = ""
            break
        if choice == "n":
            name = input_fn("Novo nome: ").strip()
            if not name:
                output("O nome não pode ficar vazio.")
                continue
            data.at[index, "relevance"] = "relevante"
            data.at[index, "selected_name_option"] = "manual"
            data.at[index, "manual_name"] = name
            break
        if choice == "o":
            edit_note(data, index, annotator_id, personal_path, input_fn, output)
            continue
        if choice in {"b", "q"}:
            return choice
        output("Opção inválida.")
    data.at[index, "annotator_id"] = annotator_id
    data.at[index, "annotated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_save(data, personal_path)
    return "saved"


def adjudication_name_candidates(row):
    candidates = []
    for prefix in ("victoria", "gabriel"):
        if str(row.get(f"{prefix}_relevance", "")).strip().lower() != "relevante":
            continue
        name = str(row.get(f"{prefix}_selected_name", "")).strip()
        option = str(row.get(f"{prefix}_name_option", "")).strip().lower()
        if name and name not in [candidate[0] for candidate in candidates]:
            candidates.append((name, option))
    return candidates


def annotate_adjudication_name(data, index, annotator_id, personal_path, input_fn, output):
    candidates = adjudication_name_candidates(data.iloc[index])
    output("\nNomes escolhidos pelos anotadores anteriores (sem identificar o anotador)")
    for number, (name, _) in enumerate(candidates, 1):
        output(f"  {number}. {name}")
    allowed = "/".join(str(i) for i in range(1, len(candidates) + 1))
    while True:
        choice = input_fn(f"Nome [{allowed}, n=novo, o=observação, b=voltar, q=sair]: ").strip().lower()
        if choice.isdigit() and 1 <= int(choice) <= len(candidates):
            name, original_option = candidates[int(choice) - 1]
            if original_option in {"1", "2", "3"}:
                data.at[index, "selected_name_option"] = original_option
                data.at[index, "manual_name"] = ""
            else:
                data.at[index, "selected_name_option"] = "manual"
                data.at[index, "manual_name"] = name
            break
        if choice == "n":
            name = input_fn("Novo nome: ").strip()
            if not name:
                output("O nome não pode ficar vazio.")
                continue
            data.at[index, "selected_name_option"] = "manual"
            data.at[index, "manual_name"] = name
            break
        if choice == "o":
            edit_note(data, index, annotator_id, personal_path, input_fn, output)
            continue
        if choice in {"b", "q"}:
            return choice
        output("Opção inválida.")
    data.at[index, "relevance"] = "relevante"
    data.at[index, "annotator_id"] = annotator_id
    data.at[index, "annotated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    atomic_save(data, personal_path)
    return "saved"


def validate_adjudication_structure(data):
    required = {
        "disagreement_type", "victoria_relevance", "victoria_name_option", "victoria_selected_name",
        "gabriel_relevance", "gabriel_name_option", "gabriel_selected_name",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"O CSV de adjudicação não contém as colunas necessárias: {missing}")
    invalid = sorted(set(data["disagreement_type"]) - {"name", "relevance"})
    if invalid:
        raise ValueError(f"Tipos de divergência inválidos: {invalid}")


def run_session(data, personal_path, annotator_id, input_fn=input, output=print, adjudication=False):
    if adjudication:
        validate_adjudication_structure(data)
    last = last_annotation(data)
    if last is None:
        output("Nenhuma anotação anterior encontrada.")
        current = next_pending(data)
    else:
        output(f"Última anotação: tópico {last.topic_id}, em {last.annotated_at or 'horário desconhecido'}.")
        current = next_pending(data, int(last.name))
    if current is None:
        errors = validate_completed(data)
        if errors:
            raise ValueError("\n".join(errors))
        output(f"Todos os {len(data)} tópicos já foram anotados.")
        return

    history = []
    skipped = set()
    while current is not None:
        completed = int(data["relevance"].ne("").sum())
        row = data.iloc[current]
        show_cluster(row, completed, len(data), output, adjudication)
        if adjudication and row.disagreement_type == "name":
            output("\nOs dois anotadores consideraram este tópico relevante, mas escolheram nomes diferentes.")
            result = annotate_adjudication_name(data, current, annotator_id, personal_path, input_fn, output)
            if result == "q":
                atomic_save(data, personal_path)
                output(f"Progresso salvo em {personal_path}")
                return
            if result == "b":
                if history:
                    current = history.pop()
                else:
                    output("Não há tópico anterior nesta sessão.")
                continue
            history.append(current)
            following = next_pending(data, current, skipped)
            if following is None and data["relevance"].eq("").any():
                skipped.clear()
                following = next_pending(data, current)
                output("Os tópicos restantes haviam sido pulados; retornando a eles.")
            current = following
            continue
        choice = input_fn("Relevância [1=relevante, 0=irrelevante, s=pular, b=voltar, o=observação, q=sair]: ").strip().lower()
        if choice == "q":
            atomic_save(data, personal_path)
            output(f"Progresso salvo em {personal_path}")
            return
        if choice == "o":
            edit_note(data, current, annotator_id, personal_path, input_fn, output)
            continue
        if choice == "b":
            if history:
                current = history.pop()
            else:
                output("Não há tópico anterior nesta sessão.")
            continue
        if choice == "s":
            skipped.add(current)
        elif choice == "0":
            data.at[current, "relevance"] = "irrelevante"
            data.at[current, "selected_name_option"] = ""
            data.at[current, "manual_name"] = ""
            data.at[current, "annotator_id"] = annotator_id
            data.at[current, "annotated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            atomic_save(data, personal_path)
        elif choice == "1":
            if adjudication:
                result = annotate_adjudication_name(data, current, annotator_id, personal_path, input_fn, output)
            else:
                result = annotate_relevant(data, current, annotator_id, personal_path, input_fn, output)
            if result == "q":
                atomic_save(data, personal_path)
                output(f"Progresso salvo em {personal_path}")
                return
            if result == "b":
                if history:
                    current = history.pop()
                else:
                    output("Não há tópico anterior nesta sessão.")
                continue
        else:
            output("Opção inválida.")
            continue

        history.append(current)
        following = next_pending(data, current, skipped)
        if following is None and data["relevance"].eq("").any():
            skipped.clear()
            following = next_pending(data, current)
            output("Os tópicos restantes haviam sido pulados; retornando a eles.")
        current = following

    errors = validate_completed(data)
    if errors:
        raise ValueError("\n".join(errors))
    output(f"Anotação concluída: {personal_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Anotação interativa dos tópicos de gravidez")
    parser.add_argument("--annotator-id", help="ID do anotador; se omitido, será solicitado")
    parser.add_argument("--base-csv", type=Path, default=DEFAULT_BASE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adjudication", action="store_true", help="Mostra somente as decisões conflitantes anteriores")
    return parser.parse_args()


def main():
    args = parse_args()
    annotator_id = (args.annotator_id or input("ID do anotador: ")).strip()
    validate_annotator_id(annotator_id)
    personal_path = args.output_dir / f"topic_annotation_{annotator_id}.csv"
    with annotator_lock(args.output_dir, annotator_id):
        data = load_or_create(args.base_csv, personal_path, annotator_id)
        run_session(data, personal_path, annotator_id, adjudication=args.adjudication)


if __name__ == "__main__":
    main()
