# Challenge AI: Detección de Fraudes en Transacciones Bancarias

## Objetivo

Se desea abordar la detección de fraudes en transacciones bancarias utilizando **AWS Bedrock** y **Amazon SageMaker**.

1. Se les proporcionará un dataset (`credir_risk_reto.csv`) para generar descripciones relacionadas (`description`) con el riesgo crediticio mediante modelos generativos de AWS Bedrock como **GPT** o **Llama**.
2. Clasificar las descripciones generadas con Bedrock, asignando etiquetas de riesgo (`target`): “**bad risk**” o “**good risk**”.
3. Entrenar un modelo de clasificación en **Amazon SageMaker** usando las etiquetas generadas, optimizando los hiperparámetros para obtener el mejor rendimiento.
4. Se deberá desplegar el modelo en **SageMaker** para procesar nuevas transacciones y monitorear su desempeño.

> **Nota:** El sistema debe ser eficiente, escalable y capaz de adaptarse a nueva data para mantener su precisión y efectividad.

---

## Entregables

1. **Scripts funcionales para todo el flujo:**
   - a. Generación de descripciones con Bedrock  
   - b. Clasificación inicial con Bedrock  
   - c. Entrenamiento del modelo supervisado en SageMaker  
   - d. Despliegue del modelo y pruebas de inferencia  
   - e. El código debería estar alojado en un repositorio con instrucciones claras (deseable)

2. **Informe corto:**
   - a. Descripción del flujo de trabajo  
   - b. Explicación de decisiones técnicas  
   - c. Métricas de desempeño del modelo  
   - d. Screenshots o logs de Bedrock y SageMaker  

3. Enlace del endpoint desplegado en SageMaker o evidencia funcional (video o capturas de pantalla).  
4. Script o instrucciones para realizar consultas al modelo.  
5. Dataset modificado con las nuevas columnas (`description` y `target`).  
6. Incluir un video presentando sus hallazgos y/o prototipo.

---

## Diccionario de datos

El dataset contiene **1000 entradas** con **9 columnas** categóricas/numéricas. Cada entrada representa a una persona como **buen** o **mal** riesgo crediticio de acuerdo con el set de características. Cada columna representa lo siguiente:

- **Age:** Edad de la persona  
- **Sex:** Sexo de la persona  
- **Job:**  
  - 0: unskilled and non-resident  
  - 1: unskilled and resident  
  - 2: skilled  
  - 3: highly skilled  
- **Housing:** Tipo de alojamiento  
- **Saving accounts:** Tipo de cuenta de ahorro  
- **Checking account:** Tipo de cuenta corriente  
- **Credit amount:** Monto de crédito  
- **Duration (meses):** Tiempo de préstamo  
- **Purpose:** Motivo del préstamo  

---

## Entrega

Enviar un correo con la url de la solución del ejercicio en tu repo a **ai@reevalua.com**
Usa el asunto **Ejercicio Challenge AI**.
