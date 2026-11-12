"""Kaggle CLI for a Claude cloud session, where the token is injected by the proxy.

In the cloud session the real Kaggle token is a network secret: the session's
proxy writes it into the Authorization header of every request to
api.kaggle.com. The CLI, before any command, "introspects" the token by
sending it in the request *body*, where the proxy cannot fill it in, so the
check fails with a placeholder. This wrapper skips that one check; every
real API call still carries the real token in the header.

    KAGGLE_API_TOKEN=KGAT_placeholder python3 -I experiments/kaggle_cli.py kernels status ommnaidu/gemma-e2b-lora-evidence

Run it from outside the repo (e.g. `cd /tmp`) or with `-I`, so a local file
cannot shadow a library module. On a normal machine with a real token in
KAGGLE_API_TOKEN, use the plain `kaggle` command instead.
"""

import sys

from kaggle.api.kaggle_api_extended import KaggleApi

KAGGLE_USER = "ommnaidu"
KaggleApi._introspect_token = lambda self, token: KAGGLE_USER

from kaggle.cli import main  # noqa: E402

sys.argv[0] = "kaggle"
main()
