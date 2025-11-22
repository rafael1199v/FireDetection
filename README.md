# FireDetectionBot

Esta es una aplicación creada en python que contiene un modelo de machine learning para la detección de incendios forestales en el departamento de Santa Cruz. Además los reportes analizados se los envia por un bot de telegram para su debido análisis por la estación mas cercana a ese punto.

# Dependencias
- `private_key.json`: Archivo de llave privada
- `.env`: Archivo con las variables de entorno

Todas las dependencias se integran en la carpeta `config`

# Instalación 

Crea un entorno virtual
```
python -m venv .venv
```

Inicia el entorno virtual
```
./.venv/Scripts/activate
```

Instale las dependencias necesarias
```
pip install -r requirements.txt
```

# Ejecución

Ejecuta el bot de telegram con:
```
python main.py
```


Para entrenar el modelo se puede ejecutar con:
```
python ./fireDetectionApp/models/model.py
```

En caso de tener errores, en los comandos. Revisa el separador de las rutas de directorios dependiendo del sistema operativo. Además es necesario tener un directorio donde se tengan los datos para el entrenamiento. Ejemplo `data/datos.csv`

---
