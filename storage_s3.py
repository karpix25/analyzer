import logging
import os
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config import S3_ACCESS_KEY, S3_BUCKET, S3_ENDPOINT_URL, S3_REGION, S3_SECRET_KEY, USE_S3

logger = logging.getLogger("app.storage_s3")


class S3Storage:
    """S3 storage handler for uploading and managing result files."""

    def __init__(self):
        self.enabled = USE_S3
        self.bucket = S3_BUCKET
        self.region = S3_REGION

        if not self.enabled:
            logger.info("[S3] S3 storage disabled (USE_S3=false)")
            return

        if not self.bucket:
            logger.warning("[S3] S3 enabled but S3_BUCKET not configured")
            self.enabled = False
            return

        # Initialize S3 client
        try:
            session_kwargs = {
                "aws_access_key_id": S3_ACCESS_KEY,
                "aws_secret_access_key": S3_SECRET_KEY,
                "region_name": self.region,
            }

            if S3_ENDPOINT_URL:
                session_kwargs["endpoint_url"] = S3_ENDPOINT_URL
                logger.info(f"[S3] Using custom endpoint: {S3_ENDPOINT_URL}")

            self.s3_client = boto3.client("s3", **session_kwargs)
            logger.info(f"[S3] Initialized S3 client for bucket: {self.bucket}")

        except Exception as e:
            logger.error(f"[S3] Failed to initialize S3 client: {e}")
            self.enabled = False

    def upload_file(self, local_path: Path, s3_key: str) -> bool:
        """Upload a file to S3.

        Args:
            local_path: Path to local file
            s3_key: S3 object key (path in bucket)

        Returns:
            True if upload successful, False otherwise
        """
        if not self.enabled:
            return False

        if not local_path.exists():
            logger.error(f"[S3] File not found: {local_path}")
            return False

        try:
            # Determine content type
            content_type = self._get_content_type(local_path)

            # Upload file
            self.s3_client.upload_file(
                str(local_path),
                self.bucket,
                s3_key,
                ExtraArgs={"ContentType": content_type, "ACL": "public-read"},
            )

            logger.info(f"[S3] Uploaded: {local_path.name} -> s3://{self.bucket}/{s3_key}")
            return True

        except ClientError as e:
            logger.error(f"[S3] Upload failed for {local_path.name}: {e}")
            return False

    def get_public_url(self, s3_key: str) -> str:
        """Get public URL for an S3 object.

        Args:
            s3_key: S3 object key

        Returns:
            Public URL for the object
        """
        if S3_ENDPOINT_URL:
            # Custom endpoint (e.g., DigitalOcean Spaces, MinIO)
            base_url = S3_ENDPOINT_URL.rstrip("/")
            return f"{base_url}/{self.bucket}/{s3_key}"
        else:
            # Standard AWS S3
            if self.region == "us-east-1":
                return f"https://{self.bucket}.s3.amazonaws.com/{s3_key}"
            else:
                return f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"

    def upload_task_results(self, task_id: str, result_dir: Path) -> dict[str, Optional[str]]:
        """Upload all result files for a task and return S3 URLs.

        Args:
            task_id: Task ID
            result_dir: Directory containing result files

        Returns:
            Dictionary mapping file types to S3 URLs (or None if upload failed)
        """
        if not self.enabled:
            return {}

        file_mapping = {
            "video_crop.mp4": "content_video",
            "clean_crop.mp4": "clean_video",
            "text_crop.mp4": "text_video",
            "frame.jpg": "content_frame",
            "text_frame.jpg": "text_frame",
            "debug.jpg": "debug_frame",
            "density_profile.jpg": "density_profile",
        }

        urls = {}

        for filename, key in file_mapping.items():
            local_path = result_dir / filename
            if local_path.exists():
                s3_key = f"results/{task_id}/{filename}"
                if self.upload_file(local_path, s3_key):
                    urls[key] = self.get_public_url(s3_key)
                else:
                    urls[key] = None
            else:
                urls[key] = None

        return urls

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type based on file extension."""
        suffix = file_path.suffix.lower()
        content_types = {
            ".mp4": "video/mp4",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webm": "video/webm",
        }
        return content_types.get(suffix, "application/octet-stream")


# Global S3 storage instance
s3_storage = S3Storage()
