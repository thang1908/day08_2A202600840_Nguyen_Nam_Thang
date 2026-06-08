"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Gợi ý nguồn:
    - https://thuvienphapluat.vn
    - https://vanban.chinhphu.vn
    - https://luatvietnam.vn

Gợi ý văn bản:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH15)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (sửa đổi 2017) - Chương XX
    - Nghị định 57/2022/NĐ-CP về danh mục chất ma tuý
"""

from pathlib import Path
import shutil
import subprocess

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOWNLOAD_TIMEOUT = (10, 30)
CURL_MAX_TIME = 180
MIN_FILE_SIZE_BYTES = 1024
CHUNK_SIZE = 1024 * 64


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")

urls = {
    "Luat_Phong_Chong_Ma_Tuy_2021.pdf":
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",

    "Luat_120_2025.pdf":
        "https://cdn.thuviennhadat.vn/upload/hinh-anh-bai-viet/DTTM/2026/luat-phong-chong-ma-tuy-2025.pdf",

    "Luat_phong_chong_ma_tuy_Nghi_Duong_Hai_Phong_2025.pdf":
        "https://cdn.haiphong.gov.vn/gov-hpg/6836/tintuc/2026/3/kh-trien-khai-thi-hanh-luat-ma-tuy-2025-ubnd.signed639082147594116577.pdf"
}


def _download_with_requests(url: str, filepath: Path) -> int:
    """Tải file bằng requests và trả về số bytes đã tải."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=DOWNLOAD_TIMEOUT,
    )
    response.raise_for_status()

    tmp_filepath = filepath.with_suffix(filepath.suffix + ".part")
    downloaded_size = 0
    with tmp_filepath.open("wb") as file:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            file.write(chunk)
            downloaded_size += len(chunk)

    tmp_filepath.replace(filepath)
    return downloaded_size


def _download_with_curl(url: str, filepath: Path) -> int:
    """Tải file bằng curl để ổn định hơn với một số CDN."""
    tmp_filepath = filepath.with_suffix(filepath.suffix + ".part")
    tmp_filepath.unlink(missing_ok=True)

    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            str(DOWNLOAD_TIMEOUT[0]),
            "--max-time",
            str(CURL_MAX_TIME),
            "-o",
            str(tmp_filepath),
            url,
        ],
        check=True,
    )

    downloaded_size = tmp_filepath.stat().st_size
    tmp_filepath.replace(filepath)
    return downloaded_size


def download_file(filename: str, url: str, overwrite: bool = False) -> Path:
    """Tải một file từ URL và lưu vào DATA_DIR."""
    filepath = DATA_DIR / filename

    if filepath.exists() and not overwrite:
        print(f"↷ Bỏ qua vì đã tồn tại: {filepath}", flush=True)
        return filepath

    if shutil.which("curl"):
        downloaded_size = _download_with_curl(url, filepath)
    else:
        downloaded_size = _download_with_requests(url, filepath)

    if downloaded_size < MIN_FILE_SIZE_BYTES:
        filepath.unlink(missing_ok=True)
        raise ValueError(
            f"File tải về quá nhỏ ({downloaded_size} bytes), có thể URL không đúng: {url}"
        )

    print(f"✓ Đã tải: {filepath} ({downloaded_size:,} bytes)", flush=True)
    return filepath


def download_all(overwrite: bool = False):
    """Tải toàn bộ văn bản pháp luật trong danh sách urls."""
    setup_directory()

    success_count = 0
    for filename, url in urls.items():
        try:
            print(f"Đang tải {filename}...", flush=True)
            download_file(filename, url, overwrite=overwrite)
            success_count += 1
        except requests.RequestException as exc:
            print(f"✗ Lỗi mạng khi tải {filename}: {exc}", flush=True)
        except subprocess.CalledProcessError as exc:
            print(f"✗ Lỗi curl khi tải {filename}: {exc}", flush=True)
        except ValueError as exc:
            print(f"✗ Lỗi dữ liệu khi tải {filename}: {exc}", flush=True)

    print(f"Hoàn tất: {success_count}/{len(urls)} file đã sẵn sàng trong {DATA_DIR}", flush=True)


if __name__ == "__main__":
    download_all()
