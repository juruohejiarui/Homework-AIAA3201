import argparse

def main():
    parser = argparse.ArgumentParser(description="AIAA3201 Lab: MLP & CNN Fundamentals")
    parser.add_argument('--part', type=int, choices=[1, 2, 3], help="Specify which part of the lab to run (1, 2, or 3).")
    args = parser.parse_args()

    if args.part == 1 or args.part is None:
        print("\n" + "="*50)
        print("Running Part 1: Numpy Backpropagation (XOR)")
        print("="*50)
        import part1_numpy_backprop
        part1_numpy_backprop.run()

    if args.part == 2 or args.part is None:
        print("\n" + "="*50)
        print("Running Part 2: Activation Functions Arena")
        print("="*50)
        import part2_pytorch_activations
        part2_pytorch_activations.run()

    if args.part == 3 or args.part is None:
        print("\n" + "="*50)
        print("Running Part 3: CNN Dimension & Parameter Calculator")
        print("="*50)
        import part3_cnn_dimensions
        part3_cnn_dimensions.run()

if __name__ == "__main__":
    main()