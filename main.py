from telegram import Update
from fireDetectionApp import app
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='config/.env')

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Servicio en ejecución y sin problemas", parse_mode="Markdown")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hola. Soy un bot que ayudará a cuidar Santa Cruz de los incendios forestales.", parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "Comandos disponibles:\n/start: Mensaje de bienvenida.\n/detect: Iniciar detección.\n/status: Verificar estado del servicio"
    await update.message.reply_text(message, parse_mode="Markdown")

async def detect_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id

    await context.bot.send_message(
        chat_id=chat_id,
        text="Iniciando proceso de deteccion"
    )

    df, images, message = app.fire_detection_app()

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
    application = ApplicationBuilder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("detect", detect_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    print("Ejecutando app")
    main()