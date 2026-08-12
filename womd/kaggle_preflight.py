# > Kaggle session gate: two checks at the two moments they bind. 
#   FIRST cell: the working copy must be the same code version as the uploaded code dataset - a stale copy once ran
#       outdated staging code for a full loop. 
#   LAST cell before Save Version: no credential file in /kaggle/working - Save Version snapshots the whole directory 
#       and would publish the GCP key into the published dataset.

import sys
from pathlib import Path

from womd import contract

# Check 1, runs FIRST. The running code states its version (imported contract = working copy); grep the MOUNT's  contract.py text 
# for that exact line. Text search, not import - two modules named contract from two paths is the exact confusion this check prevents.
def assert_working_copy_current(mounted_contract_path):
    mounted_text = Path(mounted_contract_path).read_text()
    if f'STAGING_CODE_VERSION = "{contract.STAGING_CODE_VERSION}"' not in mounted_text:
        raise SystemExit(
            f"working copy version {contract.STAGING_CODE_VERSION} not found in {mounted_contract_path}"
        )

# Check 2, runs LAST, as its own final cell. gcloud.json is written DURING the session, so a run-first check cannot see it. 
# SystemExit not assert: a Kaggle cell should die with one readable line, and -O can never strip it.
def assert_no_credentials(working_directory):
    credential_paths = list(Path(working_directory).rglob("gcloud.json"))
    if credential_paths:
        raise SystemExit(f"credential files present, do NOT Save Version: {credential_paths}")


if __name__ == "__main__":
    if sys.argv[1] == "first":
        assert_working_copy_current(sys.argv[2])
    else:
        assert_no_credentials(sys.argv[2])