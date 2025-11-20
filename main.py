from fireDetectionApp import app

def main():
    df, imagenes_guardadas = app.fire_detection_app()
    print(df, imagenes_guardadas)

if __name__ == "__main__":
    main()