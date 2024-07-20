import zstandard
import os
import json
import sys
from datetime import datetime
import logging.handlers

# Input file path (adjust as needed)
input_file = r"collegeresults_submissions.zst"

# Output file path
output_file = r"collegeresults_data.json"

# Date range
from_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
to_date = datetime.strptime("2030-12-31", "%Y-%m-%d")

# Set up logging
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)
if not os.path.exists("logs"):
    os.makedirs("logs")
log_file_handler = logging.handlers.RotatingFileHandler(os.path.join("logs", "bot.log"), maxBytes=1024*1024*16, backupCount=5)
log_file_handler.setFormatter(log_formatter)
log.addHandler(log_file_handler)

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip(), file_handle.tell()
            buffer = lines[-1]
        reader.close()

def process_file(input_file, output_file, from_date, to_date):
    log.info(f"Input: {input_file} : Output: {output_file}")
    
    file_size = os.stat(input_file).st_size
    created = None
    matched_lines = 0
    filtered_lines = 0
    total_lines = 0
    results = []

    for line, file_bytes_processed in read_lines_zst(input_file):
        total_lines += 1
        if total_lines % 100000 == 0:
            log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {total_lines:,} : {matched_lines:,} : {filtered_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj['created_utc']))

            if created < from_date:
                continue
            if created > to_date:
                continue

            selftext = obj.get("selftext", "").strip()
            link_flair_text = obj.get("link_flair_text")

            # Filter out [removed], [deleted], and null link_flair_text
            if selftext in ["[removed]", "[deleted]"] or link_flair_text is None:
                filtered_lines += 1
                continue

            matched_lines += 1
            results.append({
                "id": obj["id"],
                "link_flair_text": link_flair_text,
                "selftext": selftext
            })

        except (KeyError, json.JSONDecodeError) as err:
            log.warning(f"Error processing line: {err}")
            log.warning(line)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    log.info(f"Complete : {total_lines:,} : {matched_lines:,} : {filtered_lines:,}")

if __name__ == "__main__":
    log.info(f"From date {from_date.strftime('%Y-%m-%d')} to date {to_date.strftime('%Y-%m-%d')}")
    process_file(input_file, output_file, from_date, to_date)