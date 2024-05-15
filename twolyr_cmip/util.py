import os

hostname = os.getenv("HOSTNAME")

if hostname is not None:
    if hostname[:5] in ["crhtc", "caspe"]:
        DATADIR = "/glade/work/apauling/pythonstuff/volcano_project/data/cmip6data"
    else:
        raise NotImplementedError(f"Hostname {hostname} not yet supported")
else:
    DATADIR = "tests/testdata"