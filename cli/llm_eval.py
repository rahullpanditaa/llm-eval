import argparse
from lib.load_json import extract_info_command
from lib.generate_answer import generate_answer_command

def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    extract_json_parser = subparsers.add_parser("extract-info", help="Extract the last user query and specified context texts from json samples")
    extract_json_parser.add_argument("--conversation", type=int, choices=[1,2], help="Sample conversation json to load")
    extract_json_parser.add_argument("--k", type=int, nargs="?", default=5, help="Number of context texts to retrieve from sample context json loaded")

    generate_answer_parser = subparsers.add_parser("generate-answer", help="Generate LLM answer to last user message from selected conversation")
    generate_answer_parser.add_argument("--conversation", type=int, choices=[1,2], help="Sample conversation json to load")
    generate_answer_parser.add_argument("--k", type=int, nargs="?", default=5, help="Number of context texts to retrieve from sample context json loaded")
    
    args=parser.parse_args()

    match args.command:
        case "extract-info":
            extract_info_command(n=args.conversation, k=args.k)
        case "generate-answer":
            generate_answer_command(n=args.conversation,k=args.k)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()