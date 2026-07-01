import os
import shutil

from rag import _load_index


def main() -> None:
    persist_dir = os.getenv("PERSIST_DIR", "./storage")
    data_dir = os.getenv("DATA_DIR", "driving_data")

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    os.environ["PREBUILT_INDEX"] = "0"
    _load_index(persist_dir=persist_dir, data_dir=data_dir)
    print(f"Rebuilt index in {persist_dir}")


if __name__ == "__main__":
    main()
