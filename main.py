import datetime
from telegram import Update
from fireDetectionApp import app
from fireDetectionApp.utils.close_station import get_close_station
from fireDetectionApp.fire_stations.stations_list import stations
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv
import os
import pytz

BOLIVIA_TZ = pytz.timezone("America/La_Paz")

load_dotenv(dotenv_path='config/.env')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Servicio en ejecución y sin problemas", parse_mode="Markdown")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola. Soy un bot que ayudará a cuidar Santa Cruz de los incendios forestales.", parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "Comandos disponibles:\n/start: Mensaje de bienvenida.\n/detect: Iniciar detección.\n/status: Verificar estado del servicio"
    await update.message.reply_text(message, parse_mode="Markdown")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await update.message.reply_text("Ha ocurrido un error inesperado. Vuelve a intentarlo.")
    finally:
        print(context.error)

async def cron_job(context: ContextTypes.DEFAULT_TYPE):
    df, images, message, points_full_matrix = app.fire_detection_app(n_puntos=1, distance_m=500, points_around_number=5)
    station = get_close_station(latitudeStation=points_full_matrix[0][0][0], longitudeStation=points_full_matrix[0][0][1], stations=stations)

    await context.bot.send_message(chat_id=station.chat_id, text="Atencion!!! Hemos analizado uno de los puntos cercanos a la estacion.")
    await context.bot.send_message(chat_id=station.chat_id, text=message)

    for image in images:
        with open(image, 'rb') as img:
            await context.bot.send_photo(
                chat_id=station.chat_id,
                photo=img
            )

async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Cron job ejecutado manualmente")
    await cron_job(context=context)
    

async def detect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    await context.bot.send_message(
        chat_id=chat_id,
        text="Iniciando proceso de deteccion"
    )

    df, images, message, points_full_matrix = app.fire_detection_app(n_puntos=3, distance_m=500, points_around_number=5)

    await context.bot.send_message(
        chat_id=chat_id,
        text=message
    )

    for image in images:
        with open(image, 'rb') as img:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=img
            )

def main():
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    application = ApplicationBuilder().token(TOKEN).read_timeout(60).write_timeout(60).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("detect", detect_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("test", test_command))

    application.add_error_handler(error_handler)

    application.job_queue.run_repeating(
        callback=cron_job,
        interval=600,
        first=30,
        last= datetime.time(hour=12, minute=0, second=0, tzinfo=BOLIVIA_TZ)
    )

    print("Ejecutando app")
    application.run_polling(allowed_updates=Update.ALL_TYPES)



if __name__ == "__main__":
    main()