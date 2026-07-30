"""Port of server/routes/import.js. Named import_clips.py, not import.py --
`import` is a reserved word in Python.

Uses the imageio-ffmpeg binary (already a runtime dependency for shared/frames.py)
rather than reaching into server/node_modules — this matches the plan's own
library-choices table (section 2): "imageio-ffmpeg on the dev box; system
ffmpeg (apt) on the Pi", not a dependency on the Node install.
"""
import asyncio
import random
import re
import string

from fastapi import APIRouter, UploadFile
from fastapi.responses import JSONResponse
from imageio_ffmpeg import get_ffmpeg_exe

from app.config import RECORDINGS_DIR, SERVER_DIR
from app.lib.iso_time import filename_timestamp

router = APIRouter()

TMP_DIR = SERVER_DIR / "tmp"
MAX_FILES = 20
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB raw import

ACCEPTED_TYPES = {
    "video/mp4", "video/quicktime", "video/x-msvideo",
    "video/webm", "video/x-matroska", "video/avi",
}
ACCEPTED_EXT_RE = re.compile(r"\.(mp4|mov|avi|mkv|webm|m4v)$", re.IGNORECASE)


def _is_accepted(content_type: str | None, original_name: str) -> bool:
    return (content_type in ACCEPTED_TYPES) or bool(ACCEPTED_EXT_RE.search(original_name or ""))


def _random_suffix() -> str:
    """Matches JS's Math.random().toString(36).slice(2, 6): 4 base36-ish chars.
    Just a uniqueness suffix, not security-sensitive, so exact PRNG parity
    with JS isn't required."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=4))


async def _convert_to_webm(input_path, output_path) -> None:
    """Args copied verbatim from import.js's convertToWebm() (fluent-ffmpeg
    output options), run via asyncio.create_subprocess_exec instead of
    fluent-ffmpeg, which is just a CLI-arg builder anyway."""
    args = [
        get_ffmpeg_exe(),
        "-i", str(input_path),
        "-c:v", "libvpx-vp9",
        "-crf", "33",
        "-b:v", "0",
        "-an",
        "-t", "30",
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-f", "webm",
        "-y",
        str(output_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(stderr.decode(errors="replace")[-500:])


@router.post("")
@router.post("/")
async def import_clips(videos: list[UploadFile]):
    """multipart, field name "videos", up to 20 files at once."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if not videos:
        return JSONResponse(status_code=400, content={"error": "No files uploaded"})
    videos = videos[:MAX_FILES]

    results = []
    for file in videos:
        original_name = file.filename or ""
        if not _is_accepted(file.content_type, original_name):
            results.append({"original": original_name, "status": "error", "detail": f"Unsupported file type: {file.content_type}"})
            continue

        tmp_path = TMP_DIR / f"upload-{filename_timestamp()}-{_random_suffix()}"
        out_name = f"recording-import-{filename_timestamp()}-{_random_suffix()}.webm"
        out_path = RECORDINGS_DIR / out_name

        try:
            size = 0
            with tmp_path.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        raise ValueError("File too large (max 500MB)")
                    out.write(chunk)

            if file.content_type == "video/webm":
                tmp_path.replace(out_path)
            else:
                await _convert_to_webm(tmp_path, out_path)
                tmp_path.unlink(missing_ok=True)

            stat = out_path.stat()
            results.append({"filename": out_name, "size": stat.st_size, "original": original_name, "status": "ok"})
        except Exception as err:
            tmp_path.unlink(missing_ok=True)
            results.append({"original": original_name, "status": "error", "detail": str(err)})

    failed = [r for r in results if r["status"] == "error"]
    status_code = 500 if failed and len(failed) == len(results) else 200
    return JSONResponse(status_code=status_code, content={"results": results})
