import boto3
import json
import argparse

def query_endpoint(endpoint_name, description_text):
    """Invoca el endpoint de SageMaker y devuelve la predicción."""
    client = boto3.client('sagemaker-runtime')
    payload = {'description': description_text}
    
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        result = json.loads(response['Body'].read().decode())
        return result
    except Exception as e:
        print(f"Error al invocar el endpoint: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clasifica el riesgo crediticio usando un endpoint de SageMaker.")
    parser.add_argument(
        "description", 
        type=str, 
        help="La descripción del perfil de crédito a clasificar. Debe estar entre comillas."
    )
    parser.add_argument(
        "--endpoint-name", 
        type=str, 
        default='credit-risk-classifier-endpoint-final',
        help="El nombre del endpoint de SageMaker a invocar."
    )
    args = parser.parse_args()
    
    print(f"Consultando al endpoint: {args.endpoint_name}")
    prediction = query_endpoint(args.endpoint_name, args.description)
    
    if prediction:
        print("\n--- Resultado de la Clasificación ---")
        print(f"  Predicción: {prediction['prediction']}")
        print(f"  Confianza:  {prediction['confidence']:.4f}")
        print("------------------------------------")