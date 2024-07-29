"""
Utility functions for downloading checkpoints and configuration files.
Borrowed from jsk_data
"""

import hashlib
import os
import os.path as osp
import re
import shlex
import subprocess
import shutil
import stat
import sys
import tarfile
import zipfile
from vision_anything import CHECKPOINT_ROOT


def is_file_writable(path):
    if not os.path.exists(path):
        return True  # if file does not exist, any file is writable there
    st = os.stat(path)
    return (
        bool(st.st_mode & stat.S_IWUSR)
        and bool(st.st_mode & stat.S_IWGRP)
        and bool(st.st_mode & stat.S_IWOTH)
    )


def check_md5sum(path, md5):
    # validate md5 string length if it is specified
    if md5 and len(md5) != 32:
        raise ValueError("md5 must be 32 charactors\n" "actual: {} ({} charactors)".format(md5, len(md5)))
    path_md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    print("[%s] checking md5sum)" % path)
    is_same = path_md5 == md5

    print("[%s] Finished checking md5sum" % path)
    return is_same


def is_google_drive_url(url):
    m = re.match(r"^https?://drive.google.com/uc\?id=.*$", url)
    return m is not None


def download(client, url, output, quiet=False, chmod=True, timeout=30):
    print("[%s] Downloading from %s" % (output, url))
    if client == "wget":
        cmd = "{client} {url} -O {output} -T {timeout} --tries 1".format(
            client=client, url=url, output=output, timeout=timeout
        )
    else:
        cmd = "{client} {url} -O {output}".format(client=client, url=url, output=output)
    if quiet:
        cmd += " --quiet"
    try:
        status = subprocess.call(shlex.split(cmd))
    finally:
        if chmod:
            if not is_file_writable(output):
                os.chmod(output, 0o766)
    if status == 0:
        print("[%s] Finished downloading" % output)
    else:
        print("[%s] Failed downloading. exit_status: %d" % (output, status))


def download_data(path, url, md5, download_client=None, quiet=True, chmod=True, n_times=2):
    if download_client is None:
        if is_google_drive_url(url):
            download_client = "gdown"
        else:
            download_client = "wget"
    # cmd_exists
    if not any(
        os.access(os.path.join(path, download_client), os.X_OK)
        for path in os.environ["PATH"].split(os.pathsep)
    ):
        print(
            "\033[31mDownload client [%s] is not found. Skipping download\033[0m" % download_client,
            file=sys.stderr,
        )
        return True
    if not osp.isabs(path):
        path = osp.join(CHECKPOINT_ROOT, path)
    if not osp.exists(osp.dirname(path)):
        try:
            os.makedirs(osp.dirname(path))
        except OSError as e:
            # can fail on running with multiprocess
            if not osp.isdir(path):
                raise
    # check if cache exists, and update if necessary
    # Try n_times download.
    # https://github.com/jsk-ros-pkg/jsk_common/issues/1574
    try_download_count = 0
    while True:
        if osp.exists(path):
            if check_md5sum(path, md5):
                break
            else:
                os.remove(path)
                download(download_client, url, path, quiet=quiet, chmod=chmod)
                try_download_count += 1
                if try_download_count >= n_times:
                    path_md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
                    print(
                        "\033[31m[%s] md5sum mismatched! "
                        "expected: %s vs actual: %s\033[0m" % (path, md5, path_md5),
                        file=sys.stderr,
                    )
                    return False
        else:
            if try_download_count >= n_times:
                print(
                    "\033[31m[%s] could not download file. " "maximum download trial exceeded.\033[0m" % path,
                    file=sys.stderr,
                )
                return False
            else:
                download(download_client, url, path, quiet=quiet, chmod=chmod)
                try_download_count += 1
    return True
