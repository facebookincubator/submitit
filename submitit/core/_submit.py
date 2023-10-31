# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
from submitit.core.submission import submitit_main

if __name__ == "__main__":
    # This script is called by Executor.submit
    sys.stderr.write("submitit main\n")
    submitit_main()
