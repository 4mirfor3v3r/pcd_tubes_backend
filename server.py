import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=os.environ.get("PORT", 8000), reload=False, log_level="info", workers=1)