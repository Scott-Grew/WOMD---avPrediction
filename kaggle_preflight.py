# > Kaggle session gate: two checks at the two moments they bind. FIRST cell: the working
# copy must be the same code version as the uploaded code dataset - a stale copy once ran
# outdated staging code for a full loop. LAST cell before Save Version: no credential file
# in /kaggle/working - Save Version snapshots the whole directory and would publish the
# GCP key into the published dataset.

import womd.runtime_env
import sys
from pathlib import Path

from womd import contract

# Check 1, runs FIRST. The running code states its version (imported contract = working
# copy); search the MOUNT's contract.py text for that exact line. Text search, not import -
# importing two modules both named contract from two paths is the confusion this prevents.
def assert_working_copy_current(mounted_contract_path):
    mounted_text = Path(mounted_contract_path).read_text()
    if f'STAGING_CODE_VERSION = "{contract.STAGING_CODE_VERSION}"' not in mounted_text:
        raise SystemExit(
            f"working copy version {contract.STAGING_CODE_VERSION} not found in {mounted_contract_path}"
        )

# Check 2, runs LAST, as its own final cell. The credential file is written DURING the
# session, so a check at the start cannot see it. Content sweep, not filename match: GCP's
# own download naming defeats an exact-name check on the first deviation, and every GCP key
# carries "private_key" while OAuth credentials carry "refresh_token". SystemExit not
# assert: a Kaggle cell should die with one readable line, and python -O can never strip it.
def assert_no_credentials(working_directory):
    credential_markers = ("private_key", "refresh_token")
    credential_paths = []
    for json_path in Path(working_directory).rglob("*.json"):
        json_text = json_path.read_text(errors="ignore")
        if json_path.name == "gcloud.json" or any(marker in json_text for marker in credential_markers):
            credential_paths.append(json_path)
    if credential_paths:
        raise SystemExit(f"credential files present, do NOT Save Version: {credential_paths}")


if __name__ == "__main__":
    if sys.argv[1] == "first":
        assert_working_copy_current(sys.argv[2])
    else:
        assert_no_credentials(sys.argv[2])
