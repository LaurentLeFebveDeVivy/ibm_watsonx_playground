"""IBM Cloud Object Storage helpers for downloading documents."""

from __future__ import annotations

import tempfile
from pathlib import Path

import ibm_boto3
from ibm_botocore.client import Config

from config.config import COSCredentials, get_cos_credentials


def _build_cos_client(creds: COSCredentials):
    """Create an ibm_boto3 S3 resource authenticated via IAM."""
    return ibm_boto3.resource(
        "s3",
        ibm_api_key_id=creds.api_key,
        ibm_service_instance_id=creds.instance_id,
        config=Config(signature_version="oauth"),
        endpoint_url=creds.endpoint,
    )


def list_objects(prefix: str = "") -> list[str]:
    """Return object keys in the configured COS bucket matching *prefix*."""
    creds = get_cos_credentials()
    cos = _build_cos_client(creds)
    bucket = cos.Bucket(creds.bucket_name)
    return [obj.key for obj in bucket.objects.filter(Prefix=prefix)]


def download_objects(prefix: str = "") -> Path:
    """Download COS objects matching *prefix* to a temporary directory.

    Returns the Path of the temp directory containing the downloaded files.
    """
    creds = get_cos_credentials()
    cos = _build_cos_client(creds)
    bucket = cos.Bucket(creds.bucket_name)

    tmp_dir = Path(tempfile.mkdtemp(prefix="cos_"))

    for obj in bucket.objects.filter(Prefix=prefix):
        # Preserve sub-directory structure relative to prefix
        relative = obj.key[len(prefix):].lstrip("/")
        if not relative:
            continue
        dest = tmp_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        bucket.download_file(obj.key, str(dest))

    return tmp_dir
