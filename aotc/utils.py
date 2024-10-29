# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import dataclasses
import logging
import os
import time
import urllib.request
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Unionable:

  def __or__(self, other):
    if type(self) != type(other):
      raise TypeError("Unionable must be of the same type")
    # update `self` with the values in the new dataclass
    for field in dataclasses.fields(other):
      val = getattr(other, field.name)
      if val is not None:
        setattr(self, field.name, val)
    return self


def retry_download(url, filename, attempts=5, wait_between=10):
  """Download a file from a URL with retries."""
  if os.path.exists(filename):
    logger.info("Skipping download of %s, file already exists", filename)
    return
  logger.info("Downloading %s from %s to %s", filename, url, os.getcwd())
  for i in range(attempts):
    try:
      urllib.request.urlretrieve(url, filename)
      return
    except Exception as e:  # pylint: disable=broad-exception-caught
      if i < attempts - 1:
        logger.error(
            "Failed to download %s: %s, after %d attempts",
            url,
            repr(e),
            attempts,
        )
        time.sleep(wait_between)
      else:
        logger.error(
            "Failed to download %s: %s, trying again in %d seconds (attempt"
            " %d/%d)",
            url,
            repr(e),
            wait_between,
            i,
            attempts,
        )
        return
