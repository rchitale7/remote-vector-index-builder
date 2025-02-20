from botocore.config import Config

from repositories.blob_store import BlobStore

from core.config import Settings
from functools import cache
import boto3
from utils.connection import get_boto3_client

import logging

logger = logging.getLogger(__name__)

class S3BlobStore(BlobStore):

    def __init__(self, container_name: str, settings: Settings):
        self.client = get_boto3_client(settings.s3_role_arn, settings.s3_region, settings.s3_retries)
        self.container_name = container_name

    def read_blob(self, path: str, temp_dir: str):
        """Downloads a blob from s3"""
        pass

    def write_blob(self, path: str, temp_dir: str):
        """Writes a blob to s3"""
        pass

    # def cleanup_temp_files(self, temp_file_path: str) -> None:
    #     """Clean up temporary files"""
    #     try:
    #         if os.path.exists(temp_file_path):
    #             os.remove(temp_file_path)
    #     except Exception as e:
    #         logger.error(f"Error cleaning up temp files: {str(e)}")
    #
    # def calculate_checksum(self, file_path: str, algorithm='sha256') -> str:
    #     """Calculate checksum of a file"""
    #     hash_func = getattr(hashlib, algorithm)()
    #     with open(file_path, 'rb') as f:
    #         for chunk in iter(lambda: f.read(4096), b''):
    #             hash_func.update(chunk)
    #     return hash_func.hexdigest()
    #
    #
    # def download_normal(self, temp_dir: str) -> str:
    #     t_config = TransferConfig(
    #         multipart_threshold=32 * 1024 * 1024,
    #         max_concurrency=10,
    #         num_download_attempts=5,
    #         multipart_chunksize=10 * 1024 * 1024,
    #         max_io_queue=100,
    #     )
    #
    #     temp_file_path = os.path.join(temp_dir, self.path.split('/')[-1])
    #     self.client.download_file(bucket_name, object_key, temp_file_path, Config=t_config)
    #     return temp_file_path
    #
    #
    # def download_chunk(self, start_byte: int,
    #                    end_byte: int, temp_dir: str) -> Dict:
    #     temp_file_path = None
    #     try:
    #         response = self.client.get_object(
    #             Bucket=bucket_name,
    #             Key=object_key,
    #             Range=f'bytes={start_byte}-{end_byte}'
    #         )
    #
    #         chunk_data = response['Body'].read()
    #         # chunk_hash = hashlib.sha256(chunk_data).hexdigest()
    #
    #         # Use context manager to ensure file is properly closed
    #         with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir) as chunk_file:
    #             temp_file_path = chunk_file.name
    #             chunk_file.write(chunk_data)
    #
    #         # Verify written file matches calculated checksum
    #         # file_hash = self.calculate_checksum(temp_file_path)
    #         # if file_hash != chunk_hash:
    #         #     # Clean up the file if checksum fails
    #         #     os.unlink(temp_file_path)
    #         #     raise ValueError(
    #         #         f"Chunk checksum mismatch for bytes {start_byte}-{end_byte}. "
    #         #         f"Expected: {chunk_hash}, Got: {file_hash}"
    #         #     )
    #
    #         return {
    #             'start_byte': start_byte,
    #             'end_byte': end_byte,
    #             'temp_file': temp_file_path,
    #             # 'checksum': chunk_hash,
    #             'size': len(chunk_data)
    #         }
    #     except Exception as e:
    #         # Clean up the temporary file if it was created
    #         self.cleanup_temp_files(temp_file_path)
    #
    #         logger.error(f"Error downloading chunk {start_byte}-{end_byte}: {str(e)}")
    #         raise
    #
    #
    #
    # def download_s3_file_in_chunks(self, temp_dir, bucket_name: str, object_key: str,
    #                                chunk_size: int = 1024*1024*10,
    #                                max_workers: int = 10) -> str:
    #     """
    #     Download a file from S3 in parallel chunks with checksum verification.
    #
    #     Args:
    #         bucket_name (str): The S3 bucket name
    #         object_key (str): The S3 object key (file path)
    #         chunk_size (int): Size of chunks to download (default 10MB)
    #         max_workers (int): Maximum number of parallel downloads
    #
    #     Returns:
    #         str: Path to the downloaded file in temp directory
    #     """
    #
    #     try:
    #         # Get object details including checksum if available
    #         logger.info(f"Bucket name: {bucket_name}, Object key: {object_key}")
    #         response = self.client.head_object(
    #             Bucket=bucket_name,
    #             Key=object_key,
    #             ChecksumMode='ENABLED'
    #         )
    #         file_size = response['ContentLength']
    #         original_checksum = response.get('ChecksumSHA256')
    #
    #
    #         # Create temp file for final output
    #         file_extension = Path(object_key).suffix
    #         with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
    #             temp_file_path = temp_file.name
    #
    #             chunk_ranges = []
    #             for start in range(0, file_size, chunk_size):
    #                 end = min(start + chunk_size - 1, file_size - 1)
    #                 chunk_ranges.append((start, end))
    #
    #             downloaded_chunks: List[Dict] = []
    #             total_bytes_downloaded = 0
    #
    #             # Download chunks in parallel
    #             with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #                 futures = []
    #
    #                 # Submit download tasks for each chunk
    #                 for start, end in chunk_ranges:
    #                     future = executor.submit(
    #                         self.download_chunk,
    #                         bucket_name,
    #                         object_key,
    #                         start,
    #                         end,
    #                         temp_dir
    #                     )
    #                     futures.append(future)
    #
    #                 # Track progress
    #                 total_chunks = len(chunk_ranges)
    #                 completed_chunks = 0
    #
    #                 # Wait for chunks to complete and track progress
    #                 for future in as_completed(futures):
    #                     try:
    #                         chunk_info = future.result()
    #                         downloaded_chunks.append(chunk_info)
    #                         completed_chunks += 1
    #                         total_bytes_downloaded += chunk_info['size']
    #
    #                         # Calculate and display progress
    #                         progress = (total_bytes_downloaded / file_size) * 100
    #                         logger.info(
    #                             f"Progress: {progress:.2f}% "
    #                             f"({total_bytes_downloaded}/{file_size} bytes) "
    #                             f"Chunks: {completed_chunks}/{total_chunks}"
    #                         )
    #                     except Exception as e:
    #                         logger.error(f"Chunk download failed: {str(e)}")
    #                         raise
    #
    #             # Sort chunks by start_byte to maintain order
    #             downloaded_chunks.sort(key=lambda x: x['start_byte'])
    #
    #             # Combine chunks into final file
    #             logger.info("Combining chunks into final file...")
    #             # final_hash = hashlib.sha256()
    #
    #             with open(temp_file_path, 'wb') as final_file:
    #                 for chunk_info in downloaded_chunks:
    #                     with open(chunk_info['temp_file'], 'rb') as chunk_file:
    #                         chunk_data = chunk_file.read()
    #                         final_file.write(chunk_data)
    #                         # final_hash.update(chunk_data)
    #                     # Clean up chunk file
    #                     os.remove(chunk_info['temp_file'])
    #
    #             # Verify final checksum
    #             # final_checksum = final_hash.hexdigest()
    #             # if original_checksum:
    #             #     if final_checksum != original_checksum:
    #             #         raise ValueError(
    #             #             f"Final file checksum mismatch. "
    #             #             f"Expected: {original_checksum}, Got: {final_checksum}"
    #             #         )
    #             #     logger.info("Final checksum verification successful")
    #
    #
    #             logger.info(f"Download completed: {temp_file_path}")
    #             return temp_file_path
    #
    #     except botocore.exceptions.ClientError as e:
    #         error_code = e.response['Error']['Code']
    #         if error_code == 'NoSuchKey':
    #             logger.info(f"The object {object_key} does not exist in bucket {bucket_name}")
    #         elif error_code == 'NoSuchBucket':
    #             logger.info(f"The bucket {bucket_name} does not exist")
    #         else:
    #             logger.info(f"Error downloading object: {traceback.format_exc()} {e}")
    #
    #         # Clean up temp files
    #         self.cleanup_temp_files(temp_dir, temp_file_path)
    #         raise
    #     except Exception as e:
    #         logger.error(f"Unexpected error: {traceback.format_exc()} {e}")
    #         # Clean up temp files
    #         self.cleanup_temp_files(temp_dir, temp_file_path)
    #         raise

