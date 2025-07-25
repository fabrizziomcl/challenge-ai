import torch
import json
import os
from transformers import DistilBertTokenizer, DistilBertModel

class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        return self.classifier(pooler)

def model_fn(model_dir):
    """Carga el modelo y el tokenizador desde los artefactos guardados."""
    print("Cargando el modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = DistilBERTClass().to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model_best.bin'), map_location=device))
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    
    model.eval()
    print("Modelo cargado exitosamente.")
    return {'model': model, 'tokenizer': tokenizer, 'device': device}

def input_fn(request_body, request_content_type):
    """Parsea los datos de entrada. Espera un JSON con una clave 'description'."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if 'description' not in data:
            raise ValueError("El JSON de entrada debe contener una clave 'description'")
        return data['description']
    raise ValueError(f"Tipo de contenido no soportado: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Realiza la predicción sobre el texto de entrada."""
    print("Realizando predicción...")
    tokenizer = model_dict['tokenizer']
    model = model_dict['model']
    device = model_dict['device']
    
    encoded_input = tokenizer.encode_plus(
        input_data,
        add_special_tokens=True,
        max_length=64, 
        padding='max_length',
        return_token_type_ids=True,
        truncation=True,
        return_tensors='pt' 
    )
    
    ids = encoded_input['input_ids'].to(device, dtype=torch.long)
    mask = encoded_input['attention_mask'].to(device, dtype=torch.long)
    
    with torch.no_grad():
        output = model(ids, mask)
        
    prob = torch.sigmoid(output).item()
    prediction = "good risk" if prob > 0.5 else "bad risk"
    
    print(f"Predicción: {prediction}, Confianza: {prob:.4f}")
    return {'prediction': prediction, 'confidence': prob}

def output_fn(prediction_output, accept):
    """Formatea la salida de la predicción a JSON."""
    return json.dumps(prediction_output), accept