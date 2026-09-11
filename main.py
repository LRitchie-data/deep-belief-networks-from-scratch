from src.dnn import DeepClassifier

def main():
    print("Initialisation du Deep Belief Network...")
    model = DeepClassifier(layer_sizes=[784, 200, 200], n_classes=10)
    print("Architecture configuree : [784 -> 200 -> 200 -> 10]")
    print("Pret pour l'entrainement et le fine-tuning.")

if __name__ == "__main__":
    main()
