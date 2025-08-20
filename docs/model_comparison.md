# Comparación y Evolución de Modelos de Redes Neuronales

## Introducción

Este documento traza la evolución de los modelos de redes neuronales desarrollados para predecir el aumento del nivel del mar a partir de las emisiones de gases de efecto invernadero (GEI). El objetivo es documentar cada iteración, comparar su rendimiento y justificar los cambios metodológicos realizados en cada paso, pasando de un prototipo exploratorio a un modelo más robusto y fiable.

---

## Modelo V1: Baseline Inicial (Desde `2.johanna/Project`)

La primera versión del modelo fue un intento de establecer una línea base simple para el problema.

*   **Versión:** 1.0
*   **Arquitectura:**
    *   **Capas:** 0 capas ocultas. Es funcionalmente una regresión lineal multivariable.
    *   **Neuronas:** N/A
    *   **Funciones de Activación:** Lineal en la capa de salida.
*   **Parámetros de Entrenamiento:**
    *   **Optimizador:** Descenso de Gradiente por Lotes (Batch Gradient Descent).
    *   **Tasa de Aprendizaje:** Fija (e.g., 0.002), ajustada manualmente.
    *   **Épocas:** Hasta 1,000,000.
*   **Métricas de Rendimiento:**
    *   **MSE (Entrenamiento):** Muy bajo, a menudo cercano a 0.008.
    *   **Advertencia:** Estas métricas se obtuvieron utilizando una **división de datos aleatoria**, que no es apropiada para series temporales. Los resultados son, por lo tanto, **engañosamente optimistas** y no reflejan la verdadera capacidad predictiva del modelo.
*   **Conclusión:** Este modelo sirvió como un buen primer paso para implementar el flujo de datos. Sin embargo, su simplicidad y, lo que es más importante, su método de evaluación defectuoso, lo hacen inadecuado para una predicción fiable. Demostró que el modelo podía "memorizar" los datos barajados, pero no que pudiera generalizar.

---

## Modelo V2: Primer Intento No Lineal (Desde `2.johanna/Project_v2`)

Esta versión intentó introducir complejidad para capturar patrones más allá de una simple línea.

*   **Versión:** 2.0
*   **Arquitectura:**
    *   **Capas:** 2 capas ocultas.
    *   **Neuronas:** 4 en la primera capa oculta, 2 en la segunda.
    *   **Funciones de Activación:** **Lineal (o ReLU en algunos experimentos)**. El uso de funciones lineales en las capas ocultas fue un **fallo conceptual clave**, ya que una secuencia de capas lineales es matemáticamente equivalente a una sola capa lineal, anulando el beneficio de la profundidad.
*   **Parámetros de Entrenamiento:**
    *   **Optimizador:** Descenso de Gradiente por Lotes.
    *   **Tasa de Aprendizaje:** Se experimentó con tasas de aprendizaje dinámicas (que disminuyen con el tiempo).
    *   **Épocas:** ~5,000 - 10,000.
*   **Métricas de Rendimiento:**
    *   **MSE:** Similar al modelo base, sin una mejora clara, lo cual es de esperar dado el problema de la función de activación.
*   **Conclusión:** Aunque la intención de añadir profundidad era correcta, la implementación no logró crear un modelo no lineal efectivo. Este intento fue valioso porque demostró la importancia de elegir las funciones de activación correctas y expuso la necesidad de un enfoque más estructurado para el modelado.

---

## Modelo V3: Enfoque Metodológico Sólido (Desde `proyecto_clean`)

Esta versión representa una refactorización completa, aplicando las mejores prácticas para obtener una evaluación fiable y un modelo más potente.

*   **Versión:** 3.0
*   **Arquitectura:**
    *   **Capas:** 2 capas ocultas.
    *   **Neuronas:** 8 en la primera capa oculta, 4 en la segunda (mayor capacidad que V2).
    *   **Funciones de Activación:** **ReLU** en ambas capas ocultas, Lineal en la capa de salida. Este es el cambio más importante, permitiendo al modelo aprender relaciones no lineales.
*   **Parámetros de Entrenamiento:**
    *   **Optimizador:** Descenso de Gradiente por Lotes.
    *   **Tasa de Aprendizaje:** Fija (0.001).
    *   **Épocas:** 20,000.
    *   **División de Datos:** **Cronológica** (Entrenamiento hasta 2000, Validación 2001-2007, Prueba 2008-2014).
    *   **Reproducibilidad:** Se fijó una semilla aleatoria (`np.random.seed(42)`).
*   **Métricas de Rendimiento (Resultados de la ejecución simulada):**
    *   **MSE Entrenamiento:** ~0.0089
    *   **MSE Validación:** ~0.1225
    *   **MSE Prueba:** ~0.2113
*   **Conclusión:** Este modelo **supera al modelo base lineal en el conjunto de prueba**, lo que confirma nuestra hipótesis de que la relación en los datos es no lineal. Sin embargo, la gran diferencia entre el MSE de entrenamiento y el de prueba indica un claro **sobreajuste**. El modelo está memorizando los datos históricos. Este es ahora el principal problema a resolver.

---

## Tabla Comparativa de Rendimiento en el Conjunto de Prueba

| Versión del Modelo | Arquitectura | División de Datos | MSE de Prueba (Normalizado) | Conclusión Clave |
| :--- | :--- | :--- | :--- | :--- |
| V1 (Baseline) | 0 Capas Ocultas | Aleatoria | ~0.01 - 0.15 | Evaluación poco fiable. |
| V2 (2-Layer Lineal) | 2 Capas Ocultas (Lineal) | Aleatoria | Similar a V1 | No se logró la no linealidad. |
| V3 (2-Layer ReLU) | 2 Capas Ocultas (ReLU) | **Cronológica** | **~0.2113** | **Evaluación fiable; sobreajuste identificado.** |

## Conclusión General y Hoja de Ruta

La evolución del proyecto demuestra un ciclo de desarrollo maduro: se comenzó con un prototipo simple (V1), se intentó una mejora que reveló una laguna conceptual (V2), y finalmente se implementó un modelo metodológicamente sólido (V3) que proporciona una **línea de base fiable y un diagnóstico claro**. 

El problema principal ha pasado de ser la *evaluación incorrecta* a ser el *sobreajuste*. Los próximos pasos deben centrarse en técnicas para mejorar la generalización del modelo:

1.  **Implementar Regularización:** Añadir regularización L2 (weight decay) para penalizar los pesos grandes.
2.  **Usar Mini-Batch Gradient Descent:** Para un entrenamiento más eficiente y una mejor convergencia.
3.  **Búsqueda de Hiperparámetros:** Optimizar sistemáticamente la arquitectura y la tasa de aprendizaje.
4.  **Enriquecer los Datos:** El paso más impactante sería añadir más variables predictoras (features) como la temperatura global o las concentraciones de CO2.
