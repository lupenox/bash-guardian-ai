from fastapi import FastAPI
from api.routes.chat import router as chat_router  # Import chat route

app = FastAPI()

@app.get("/")  # Add root route
def root():
    return {"message": "Bash AI is online!"}

app.include_router(chat_router)  # Include the chat route

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
