# 🎨 GenAI Chat + Image

## 🌐 Live Demo

🔗 https://genaiprojects-zrutgnqvv822ytx6xig2rh.streamlit.app

A minimal Streamlit chat app where the assistant can respond with text **and**
decide, on its own, when to generate an image to go along with its answer.

## How it works

1. Chat messages are stored in `st.session_state.messages`.
2. On every user input:
   - The full conversation history is sent to **Llama** running on
     [Groq](https://groq.com)'s cloud API (fast inference, no local install).
   - If the model's reply contains a tag of the form `[IMAGE: <description>]`,
     that description is extracted and sent to an image-generation API.
   - The `[IMAGE: ...]` tag itself is stripped from the visible reply text.
3. Both the text reply and (if any) the generated image are appended back
   into `st.session_state.messages`.
4. The page re-renders, showing the full conversation — text bubbles and
   inline images.

## Tech stack

| Component        | Choice                                                  |
|-------------------|----------------------------------------------------------|
| LLM               | Llama 3.3 70B via [Groq API](https://console.groq.com)   |
| Image generation  | [Pollinations.ai](https://pollinations.ai) free API (default, no key needed), or OpenAI's DALL-E 3 if `OPENAI_API_KEY` is set |
| UI                | Streamlit                                                |

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key
Sign up at [console.groq.com/keys](https://console.groq.com/keys) and create
a key.

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Paste your API key into the sidebar
No environment variables required — just paste your `GROQ_API_KEY` into the
**Groq API Key** field in the sidebar when the app opens. It's stored only in
that browser session's `st.session_state`, not written to disk.

If you'd rather not paste it every time, you can still set it as an
environment variable before launching and it will pre-fill the field:
```bash
export GROQ_API_KEY="gsk_..."
streamlit run app.py
```

### 5. (Optional) Use OpenAI's DALL-E instead of the free image API
Paste an `OPENAI_API_KEY` into the second sidebar field, or set it as an env
var the same way. If left blank, images are generated for free via
Pollinations.ai instead.

### 6. (Optional) Change the Groq model
```bash
export GROQ_MODEL="llama-3.1-8b-instant"   # faster/cheaper alternative
```
Default is `llama-3.3-70b-versatile`.

## Example prompts to try

- `"Hi, how are you?"` — plain text reply, no image.
- `"Can you draw a red fox sitting in a snowy forest?"` — the model should
  emit an `[IMAGE: ...]` tag, triggering image generation.
- `"Show me what a cyberpunk city street at night looks like"` — another
  image-triggering prompt.
- `"What's the capital of France?"` — plain text, no image.

## Environment variables

| Variable         | Default                       | Purpose                                      |
|-------------------|--------------------------------|-------------------------------------------------|
| `GROQ_API_KEY`    | *(required)*                  | Your Groq API key                                |
| `GROQ_MODEL`      | `llama-3.3-70b-versatile`     | Which Groq-hosted Llama model to chat with       |
| `OPENAI_API_KEY`  | *(unset)*                     | If set, uses DALL-E 3 instead of Pollinations    |

## Notes

- If `GROQ_API_KEY` isn't set, the app shows a friendly warning in the chat
  instead of crashing.
- Groq's free tier has generous but rate-limited usage — if you see a 429
  error, wait a moment and try again, or switch to a smaller model like
  `llama-3.1-8b-instant`.
- The model decides for itself when an image is warranted — it won't generate
  one for every message, only when its own reply includes the `[IMAGE: ...]` tag.
