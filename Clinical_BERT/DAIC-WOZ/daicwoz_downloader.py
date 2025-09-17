import asyncio, aiohttp, aiofiles, zipfile
from pathlib import Path

BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz"
OUT_DIR = Path("zips")
EXTRACT_DIR = Path("data")
START, END = 400, 493
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Encoding": "identity",
    "Connection": "close",
}

async def download_and_unzip(session, sem, filename):
    url = f"{BASE_URL}/{filename}"
    dest = OUT_DIR / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    async with sem:
        # Download
        try:
            async with session.get(url) as r:
                if r.status == 404:
                    print(f"[404] {filename}")
                    return
                r.raise_for_status()
                async with aiofiles.open(dest, "wb") as f:
                    async for chunk in r.content.iter_chunked(1 << 16):
                        await f.write(chunk)
        except Exception as e:
            print(f"[err] Failed to download {filename}: {e}")
            return

        # Validate ZIP
        try:
            with zipfile.ZipFile(dest, "r") as zf:
                if zf.testzip() is not None:
                    print(f"[bad] Corrupt ZIP: {filename}")
                    dest.unlink(missing_ok=True)
                    return
        except zipfile.BadZipFile:
            print(f"[bad] Not a valid ZIP: {filename}")
            dest.unlink(missing_ok=True)
            return

        # Unzip immediately
        extract_to = EXTRACT_DIR / dest.stem
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(extract_to)
        print(f"[ok ] Downloaded and extracted {filename} -> {extract_to}")

async def main():
    files = [f"{i}_P.zip" for i in range(START, END + 1)]
    sem = asyncio.Semaphore(2)  # gentle concurrency
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)

    async with aiohttp.ClientSession(timeout=timeout, headers=HEADERS) as session:
        await asyncio.gather(*[download_and_unzip(session, sem, f) for f in files])

if __name__ == "__main__":
    asyncio.run(main())
