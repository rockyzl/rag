from openai import AsyncOpenAI
import asyncio
from datetime import datetime
import logging
import PyPDF2  # Add this library for PDF text extraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatSession:
    def __init__(self, model="gpt-4o", max_history=10, timeout=30, pdf_path=None):
        self.client = AsyncOpenAI()
        self.model = model
        self.max_history = max_history
        self.timeout = timeout
        self.pdf_content = self._extract_pdf_content(pdf_path) if pdf_path else None
        self.system_prompt = self._create_system_prompt()

    def _extract_pdf_content(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                for page in reader.pages:
                    content += page.extract_text()
                logger.info("PDF content successfully extracted.")
                return content
        except Exception as e:
            logger.error(f"Error extracting PDF content: {str(e)}")
            return None

    def _create_system_prompt(self):
        template = """You are Bobby, a virtual assistant created by Huajun. Today is {today}.
        Refer to the following document for context when answering questions:
        {document_content}
        """
        document_content = self.pdf_content if self.pdf_content else "No additional document provided."
        return template.format(today=datetime.today().strftime('%Y-%m-%d'), document_content=document_content)

    def trim_history(self, history):
        return history[-self.max_history:] if len(history) > self.max_history else history

    async def chat_func(self, history):
        try:
            result = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}] + self.trim_history(history),
                    max_tokens=256,
                    temperature=0.5,
                    stream=True,
                ),
                timeout=self.timeout,
            )

            buffer = []
            async for r in result:
                if next_token := r.choices[0].delta.content:
                    print(next_token, flush=True, end="")
                    buffer.append(next_token)

            print("\n", flush=True)
            return "".join(buffer)

        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return "Request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error during chat: {str(e)}")
            return f"Error: {str(e)}"

    async def start(self):
        history = []
        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() == "exit":
                    break
                if not user_input:
                    continue

                history.append({"role": "user", "content": user_input})
                bot_response = await self.chat_func(history)
                history.append({"role": "assistant", "content": bot_response})

            except KeyboardInterrupt:
                print("\nExiting chat...")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")

if __name__ == "__main__":
    # Provide the path to the uploaded PDF file
    pdf_path = '../data/paper.pdf' 
    chat = ChatSession(pdf_path=pdf_path)
    asyncio.run(chat.start())
