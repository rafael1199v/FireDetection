# FireDetectionBot

Esta es una aplicación creada en python que contiene un modelo de machine learning para la detección de incendios forestales en el departamento de Santa Cruz. Además los reportes analizados se los envia por un bot de telegram para su debido análisis por la estación mas cercana a ese punto.

# Dependencias
- `private_key.json`: Archivo de llave privada. Este archivo se obtiene creando una cuenta de servicio de google para poder iniciar sesión de manera automatizada e integrarlo en sus aplicaciones. 

- `.env`: Archivo con las variables de entorno. Esta contiene tanto las ubicaciones de las variables de entorno como el token del bot de telegram para enviar mensajes.

Todas las dependencias se integran en la carpeta `config`

Esquema del `.env`

```
SERVICE_ACCOUNT_GEE=account_gee
PRIVATE_KEY_FILE=private_key_file_name
CONFIGURATION_PATH=configuration_path
TELEGRAM_TOKEN=telegram_token
```

> La cuenta de servicio de google puede obtenerla de [Crear cuenta de servicio de Google](https://docs.cloud.google.com/iam/docs/service-accounts-create?hl=es-419)

> El token del bot de telegram se obtiene por medio del `BotFather`. [Mas información aquí](https://core.telegram.org/bots/tutorial)

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


# Estructura del proyecto
```
.
├── .venv/
├── data/
├── config/
├── fireDetectionApp/
├── saved_models/
├── .env.example
├── .gitignore
├── main.py
├── README.md
└── requirements.txt
```