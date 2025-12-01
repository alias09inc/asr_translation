import argparse

from app.services.translation_service import asr_and_translate_en_to_ja


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="path to 16kHz mono wav file")
    args = parser.parse_args()

    en_text, ja_text = asr_and_translate_en_to_ja(args.audio)

    print("=== ASR (EN) ===")
    print(en_text)
    print("\n=== Translation (JA) ===")
    print(ja_text)


if __name__ == "__main__":
    main()
