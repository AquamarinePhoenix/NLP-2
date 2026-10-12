from nlplib.config import OUTPUT_DIR

OUTPUT_DIR.mkdir(parents = True, exist_ok = True)
output_path = OUTPUT_DIR / "log.txt"

def clear_file() -> None:
    with open(output_path, 'w', encoding = "utf-8") as f:
        f.write('')

def write_file(text: str) -> None:
    with open(output_path, 'a', encoding = "utf-8") as f:
        f.write(f"{text}\n")