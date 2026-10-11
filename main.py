from src.evaluation.experiments import run_demo_experiments


def main() -> None:
    print(run_demo_experiments().to_string(index=False))


if __name__ == "__main__":
    main()
