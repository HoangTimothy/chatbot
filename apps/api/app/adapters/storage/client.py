import pathlib

from app.config import settings


class ObjectStorageClient:
    """A client interface for interacting with document storage.

    For Phase 1 local development, this defaults to mock filesystem-based storage.
    """

    def __init__(self, base_path: str = settings.OBJECT_STORAGE_LOCAL_PATH):
        self.base_path = pathlib.Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def upload_file(self, file_content: bytes, file_name: str, folder: str = "") -> str:
        """Upload file content to a destination folder inside local storage.

        Returns:
            The relative path string to be saved in the database.
        """
        dest_folder = self.base_path / folder
        dest_folder.mkdir(parents=True, exist_ok=True)

        dest_file = dest_folder / file_name
        dest_file.write_bytes(file_content)

        return str(dest_file.relative_to(self.base_path).as_posix())

    def download_file(self, relative_path: str) -> bytes:
        """Retrieve stored file contents.

        Raises:
            FileNotFoundError if the file path is not valid.
        """
        target_path = self.base_path / relative_path
        if not target_path.exists() or not target_path.is_file():
            raise FileNotFoundError(f"File not found in storage: {relative_path}")
        return target_path.read_bytes()

    def delete_file(self, relative_path: str) -> None:
        """Delete a file from storage if it exists."""
        target_path = self.base_path / relative_path
        if target_path.exists() and target_path.is_file():
            target_path.unlink()
