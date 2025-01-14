from openai import AsyncOpenAI
import asyncio
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatSession:
    def __init__(self, model="gpt-4", max_history=10, timeout=30):
        self.client = AsyncOpenAI()
        self.model = model
        self.max_history = max_history
        self.timeout = timeout
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self):
        template = """You are Bobby, a virtual assistant created by Huajun. Today is {today}."""
        return template.format(today=datetime.today().strftime('%Y-%m-%d'))

    def trim_history(self, history):
        return history[-self.max_history:] if len(history) > self.max_history else history

    async def chat_func(self, history):
        try:
            # Use asyncio.wait_for to enforce a timeout
            result = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}] + self.trim_history(history),
                    max_tokens=256,
                    temperature=0.5,
                    stream=True,
                ),
                timeout=self.timeout,  # Specify timeout here
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

    async def start(self, history=None):
        if history is None:
            history = []

        try:
            # Get user input
            user_input = input("> ").strip()

            # Exit condition
            if user_input.lower() == "exit":
                print("Goodbye!")
                return

            if not user_input:
                # Skip empty input and call start recursively
                await self.start(history)
                return

            # Append user input to history
            history.append({"role": "user", "content": user_input})

            # Get bot's response
            bot_response = await self.chat_func(history)

            # Append bot response to history
            history.append({"role": "assistant", "content": bot_response})

            # Call start recursively for the next interaction
            await self.start(history)

        except KeyboardInterrupt:
            print("\nExiting chat...")
        except Exception as e:
            logger.error(f"Error in chat loop: {str(e)}")

if __name__ == "__main__":
    chat = ChatSession()
    asyncio.run(chat.start())
