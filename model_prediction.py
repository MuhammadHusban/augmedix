import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict_symptom():
    # Load the tokenizer and create the model architecture
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define the modified model architecture
    class ModifiedBertForSequenceClassification(BertForSequenceClassification):
        def __init__(self, config):
            super().__init__(config)
            self.classifier = torch.nn.Linear(config.hidden_size, 21698)

    # Create an instance of the modified model architecture
    model = ModifiedBertForSequenceClassification.from_pretrained('bert-base-uncased')

    # Load the saved model using PyTorch's native loading function
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Read the symptom names from the text file
    with open(symptom_list_path, 'r') as file:
        symptom_names = [line.strip() for line in file]

    # Start the prediction loop
    while True:
        # Get user input
        input_text = input("Enter a statement (or 'exit' to quit): ")

        # Check if user wants to exit
        if input_text.lower() == 'exit':
            break

        # Make predictions using the loaded model
        input_ids = tokenizer.encode(input_text, add_special_tokens=True)
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            outputs = model(input_ids)

        predicted_label_index = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = symptom_names[predicted_label_index]

        print("Predicted symptom:", predicted_label)
        print()

# Example usage
model_path = 'fine_tuned_model (1).pth'
symptom_list_path = 'symptoms.txt'

predict_symptom()
