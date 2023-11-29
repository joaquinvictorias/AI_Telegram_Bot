import logging
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Enable logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

telegram_bot_token = '...'
telegram_bot_name = '...'

huggingfacehub_api_token = '...'

repo_id = 'tiiuae/falcon-7b-instruct'

llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={'temperature':0.5,
                                   'max_new_tokens':2000})

template = """
You are an artificial intelligence assistant. The assistant gives clear and short answers.

{question}

"""

prompt = PromptTemplate(template=template, input_variables=['question'])

llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

# Start command

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Starts the conversation and introduces itself."""

    await context.bot.send_message(chat_id=update.effective_chat.id,
        text='Hi! My name is Butler Bot. Ask me anything you want.\n\nSend /cancel to stop talking to me.')

# Handle responses

async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Returns the reply to user after getting reply from server."""
    
    message_type: str = update.message.chat.type
    text: str = update.message.text

    if message_type == 'group':
        if telegram_bot_name in text:
            new_text: str = text.replace(telegram_bot_name, '').strip()
            user = update.message.from_user
            logger.info('Question from %s: %s', user.first_name, new_text)
            if new_text != '':
                llm_reply = llm_chain.run(new_text)
            else:
                return   
        else:
            return
    else:
        user = update.message.from_user
        logger.info('Question from %s: %s', user.first_name, update.message.text)
        if text != '':
            llm_reply = llm_chain.run(text)
        else:
            return

    await update.message.reply_text(llm_reply)

# Cancel command

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancels and ends the conversation."""
    user = update.message.from_user
    logger.info('%s canceled the conversation.', user.first_name)
    await update.message.reply_text('Bye! I hope we can talk again some day.')

def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token
    application = Application.builder().token(telegram_bot_token).build()

    # Add conversation handler
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot_reply))
    application.add_handler(CommandHandler('cancel', cancel))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()

if __name__ == '__main__':
    main()