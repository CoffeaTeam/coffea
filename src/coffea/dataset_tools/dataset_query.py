import gzip
import json
import os
import random
from collections import defaultdict

import cmd2
import yaml
from dask.distributed import Client
from rich import print
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.tree import Tree

from . import rucio_utils
from .preprocess import preprocess


def print_dataset_query(query, dataset_list, selected, console):
    table = Table(title=f"Query: [bold red]{query}")
    table.add_column("Name", justify="left", style="cyan", no_wrap=True)
    table.add_column("Tag", style="magenta", no_wrap=True)
    table.add_column("Selected", justify="center")
    table.row_styles = ["dim", "none"]
    j = 1
    for name, conds in dataset_list.items():
        ic = 0
        ncond = len(conds)
        for c, tiers in conds.items():
            dataset = f"/{name}/{c}/{tiers[0]}"
            sel = dataset in selected
            if ic == 0:
                table.add_row(
                    name,
                    f"[bold]({j})[/bold] {c}/{'-'.join(tiers)}",
                    "[green bold]Y" if sel else "[red]N",
                    end_section=ic == ncond - 1,
                )
            else:
                table.add_row(
                    "",
                    f"[bold]({j})[/bold] {c}/{'-'.join(tiers)}",
                    "[green bold]Y" if sel else "[red]N",
                    end_section=ic == ncond - 1,
                )
            ic += 1
            j += 1

    console.print(table)


class DatasetQueryApp(cmd2.Cmd):
    prompt = "\033[1;34m" + "cms-datasets" + "\033[0m > "

    def __init__(self):
        shortcuts = cmd2.DEFAULT_SHORTCUTS
        shortcuts.update(
            {
                "L": "login",
                "Q": "query",
                "QR": "query_results",
                "R": "replicas",
                "S": "select",
                "LS": "list_selected",
                "LR": "list_replicas",
                "O": "save",
                "P": "preprocess",
            }
        )
        self.console = Console()
        self.rucio_client = None
        self.selected_datasets = []
        self.last_query = ""
        self.last_query_tree = None
        self.last_query_list = None
        self.sites_allowlist = None
        self.sites_blocklist = None
        self.sites_regex = None
        self.last_replicas_results = None

        self.replica_results = defaultdict(list)
        self.replica_results_bysite = {}
        super().__init__(shortcuts=shortcuts)

    def do_login(self, args):
        """Login to the rucio client. Optionally a specific proxy file can be passed to the command.
        If the proxy file is not specified, `voms-proxy-info` is used"""
        if args:
            self.rucio_client = rucio_utils.get_rucio_client(args[0])
        else:
            self.rucio_client = rucio_utils.get_rucio_client()

        print(self.rucio_client)
        # pprint(self.rucio_client.whoami())

    def do_whoami(self, args):
        # Your code here
        if not self.rucio_client:
            print("First [bold]login (L)[/] to the rucio server")
            return
        print(self.rucio_client.whoami())

    def do_query(self, args):
        # Your code here
        with self.console.status(f"Querying rucio for: [bold red]{args}[/]"):
            outlist, outtree = rucio_utils.query_dataset(
                args.arg_list[0],
                client=self.rucio_client,
                tree=True,
                scope="cms",  # TODO configure scope
            )
            # Now let's print the results as a tree
            print_dataset_query(args, outtree, self.selected_datasets, self.console)
            self.last_query = args
            self.last_query_list = outlist
            self.last_query_tree = outtree
        print("Use the command [bold red]select (S)[/] to selected the datasets")

    def do_query_results(self, args):
        if self.last_query_list:
            print_dataset_query(
                self.last_query,
                self.last_query_tree,
                self.selected_datasets,
                self.console,
            )
        else:
            print("First [bold red]query (Q)[/] for a dataset")

    def do_select(self, args):
        """Selected the datasets from the list of query results. Input a list of indices of "all"."""
        if not self.last_query_list:
            print("First [bold red]query (Q)[/] for a dataset")
            return

        Nresults = len(self.last_query_list)
        print("[cyan]Selected datasets:")
        if args == "all":
            indices = range(0, len(self.last_query_list))  # 1 based list
        else:
            indices = map(lambda k: int(k) - 1, args.arg_list)

        for s in indices:
            if s < Nresults:
                self.selected_datasets.append(self.last_query_list[s])
                print(f"- ({s+1}) {self.last_query_list[s]}")
            else:
                print(
                    f"[red]The requested dataset is not in the list. Please insert a position <={Nresults}"
                )

    def do_list_selected(self, args):
        print("[cyan]Selected datasets:")
        table = Table(title="Selected datasets")
        table.add_column("Index", justify="left", style="cyan", no_wrap=True)
        table.add_column("Dataset", style="magenta", no_wrap=True)
        table.add_column("Replicas selected", justify="center")
        table.add_column("N. of files", justify="center")
        for i, ds in enumerate(self.selected_datasets):
            table.add_row(
                str(i + 1),
                ds,
                "[green bold]Y" if ds in self.replica_results else "[red]N",
                str(len(self.replica_results[ds]))
                if ds in self.replica_results
                else "-",
            )
        self.console.print(table)

    def do_replicas(self, args):
        if len(args.arg_list) == 0:
            print(
                "[red] Please provide a list of indices of the [bold]selected[/bold] datasets to analyze or [bold]all[/bold] to loop on all the selected datasets"
            )
            return

        if args == "all":
            datasets = self.selected_datasets
        else:
            for index in args.arg_list:
                if index.isdigit():
                    if int(index) <= len(self.selected_datasets):
                        datasets = [self.selected_datasets[int(index) - 1]]
                    else:
                        print(
                            f"[red]The requested dataset is not in the list. Please insert a position <={len(self.selected_datasets)}"
                        )

        for dataset in datasets:
            with self.console.status(
                f"Querying rucio for replicas: [bold red]{dataset}[/]"
            ):
                (
                    outfiles,
                    outsites,
                    sites_counts,
                ) = rucio_utils.get_dataset_files_replicas(
                    dataset,
                    allowlist_sites=self.sites_allowlist,
                    blocklist_sites=self.sites_blocklist,
                    regex_sites=self.sites_regex,
                    mode="full",
                    client=self.rucio_client,
                )
                self.last_replicas_results = (outfiles, outsites, sites_counts)
            print(f"[cyan]Sites availability for dataset: [red]{dataset}")
            table = Table(title="Available replicas")
            table.add_column("Index", justify="center")
            table.add_column("Site", justify="left", style="cyan", no_wrap=True)
            table.add_column("Files", style="magenta", no_wrap=True)
            table.add_column("Availability", justify="center")
            table.row_styles = ["dim", "none"]
            Nfiles = len(outfiles)

            sorted_sites = dict(
                sorted(sites_counts.items(), key=lambda x: x[1], reverse=True)
            )
            for i, (site, stat) in enumerate(sorted_sites.items()):
                table.add_row(
                    str(i), site, f"{stat} / {Nfiles}", f"{stat*100/Nfiles:.1f}%"
                )

            self.console.print(table)
            strategy = Prompt.ask(
                "Select sites",
                choices=["round-robin", "choice", "quit"],
                default="round-robin",
            )

            files_by_site = defaultdict(list)

            if strategy == "choice":
                ind = list(
                    map(
                        int,
                        Prompt.ask("Enter list of sites index to be used").split(" "),
                    )
                )
                sites_to_use = [list(sorted_sites.keys())[i] for i in ind]
                print(f"Filtering replicas with [green]: {' '.join(sites_to_use)}")

                output = []
                for ifile, (files, sites) in enumerate(zip(outfiles, outsites)):
                    random.shuffle(sites_to_use)
                    found = False
                    # loop on shuffled selected sites until one is found
                    for site in sites_to_use:
                        try:
                            iS = sites.index(site)
                            output.append(files[iS])
                            files_by_site[sites[iS]].append(files[iS])
                            found = True
                            break  # keep only one replica
                        except ValueError:
                            # if the index is not found just go to the next site
                            pass

                    if not found:
                        print(
                            f"[bold red]No replica found compatible with sites selection for file #{ifile}. The available sites are:"
                        )
                        for f, s in zip(files, sites):
                            print(f"\t- [green]{s} [cyan]{f}")
                        return

                self.replica_results[dataset] = output

            elif strategy == "round-robin":
                output = []
                for ifile, (files, sites) in enumerate(zip(outfiles, outsites)):
                    # selecting randomly from the sites
                    iS = random.randint(0, len(sites) - 1)
                    output.append(files[iS])
                    files_by_site[sites[iS]].append(files[iS])
                self.replica_results[dataset] = output

            elif strategy == "quit":
                print("[orange]Doing nothing...")
                return

            self.replica_results_bysite[dataset] = files_by_site

            # Now let's print the results
            tree = Tree(label=f"[bold orange]Replicas for [green]{dataset}")
            for site, files in files_by_site.items():
                T = tree.add(f"[green]{site}")
                for f in files:
                    T.add(f"[cyan]{f}")
            self.console.print(tree)

    def do_allowlist_sites(self, args):
        if self.sites_allowlist is None:
            self.sites_allowlist = args.arg_list
        else:
            self.sites_allowlist += args.arg_list
        print("[green]Allowlisted sites:")
        for s in self.sites_allowlist:
            print(f"- {s}")

    def do_blocklist_sites(self, args):
        if self.sites_blocklist is None:
            self.sites_blocklist = args.arg_list
        else:
            self.sites_blocklist += args.arg_list
        print("[red]Blocklisted sites:")
        for s in self.sites_blocklist:
            print(f"- {s}")

    def do_regex_sites(self, args):
        if args.startswith('"'):
            args = args[1:]
        if args.endswith('"'):
            args = args[:-1]
        self.sites_regex = rf"{args}"
        print(f"New sites regex: [cyan]{self.sites_regex}")

    def do_sites_filters(self, args):
        if args == "":
            print("[green bold]Allow-listed sites:")
            if self.sites_allowlist:
                for s in self.sites_allowlist:
                    print(f"- {s}")

            print("[bold red]Block-listed sites:")
            if self.sites_blocklist:
                for s in self.sites_blocklist:
                    print(f"- {s}")

            print(f"[bold cyan]Sites regex: [italics]{self.sites_regex}")
        if args == "clear":
            self.sites_allowlist = None
            self.sites_blocklist = None
            self.sites_regex = None
            print("[bold green]Sites filters cleared")

    def do_list_replicas(self, args):
        if len(args.arg_list) == 0:
            print("[red]Please call the command with the index of a selected dataset")
        else:
            if int(args) > len(self.selected_datasets):
                print(
                    f"[red] Select the replica with index < {len(self.selected_datasets)}"
                )
                return
            else:
                dataset = self.selected_datasets[int(args) - 1]
                if dataset not in self.replica_results:
                    print(
                        f"[red bold]No replica info for dataset {dataset}. You need to selected the replicas with [cyan] replicas {args}"
                    )
                    return
                tree = Tree(label=f"[bold orange]Replicas for [green]{dataset}")

                for site, files in self.replica_results_bysite[dataset].items():
                    T = tree.add(f"[green]{site}")
                    for f in files:
                        T.add(f"[cyan]{f}")

                self.console.print(tree)

    def do_save(self, args):
        """Save the replica information in yaml format"""
        if not len(args):
            print("[red]Please provide an output filename and format")
            return
        format = os.path.splitext(args)[1]
        output = {}
        for fileset, files in self.replica_results.items():
            output[fileset] = {"files": files, "metadata": {}}

        with open(args, "w") as file:
            if format == ".yaml":
                yaml.dump(output, file, default_flow_style=False)
            elif format == ".json":
                json.dump(output, file, indent=2)
        print(f"[green]File {args} saved!")

    def do_preprocess(self, args):
        """Perform preprocessing for concrete fileset extraction.
        Args:  output_name [step_size] [align to file cluster boundaries] [dask cluster url]
        """
        args_list = args.arg_list
        if len(args_list) < 1:
            print(
                "Please provide an output name and optionally a step size, if you want to align to file clusters, or a dask cluster url"
            )
            return
        else:
            output_file = args_list[0]
            step_size = None
            align_to_clusters = False
            dask_url = None
        if len(args_list) >= 2:
            step_size = int(args_list[1])
        if len(args_list) >= 3:
            if args_list[2] == "True":
                align_to_clusters = True
            elif args_list[2] == "False":
                align_to_clusters = False
            else:
                raise ValueError('align_to_clusters must be either "True" or "False"')
        if len(args_list) == 4:
            dask_url = args_list[3]
        if len(args_list) > 4:
            print("preprocess accepts at most 3 commandline arguments!")
            return
        replicas = {}
        for fileset, files in self.replica_results.items():
            replicas[fileset] = {"files": {f: "Events" for f in files}, "metadata": {}}
        # init a local Dask cluster
        with self.console.status(
            "[red] Preprocessing files to extract available chunks with dask[/]"
        ):
            with Client(dask_url) as _:
                out_available, out_updated = preprocess(
                    replicas,
                    maybe_step_size=step_size,
                    align_clusters=align_to_clusters,
                    skip_bad_files=True,
                )
        from IPython import embed

        embed()
        with gzip.open(f"{output_file}_available.json.gz", "wb") as file:
            print(f"Saved available fileset chunks to {output_file}_available.json.gz")
            json.dump(out_available, file, indent=2)
        with gzip.open(f"{output_file}_all.json.gz", "wb") as file:
            print(f"Saved all fileset chunks to {output_file}_all.json.gz")
            json.dump(out_updated, file, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cli", help="Start interactive CLI for dataset discovery", action="store_true"
    )
    args = parser.parse_args()

    intro_msg = r"""[bold yellow]Welcome to the datasets discovery coffea CLI![/bold yellow]
Use this CLI tool to query the CMS datasets and to select interactively the grid sites to use for reading the files in your analysis.
Some basic commands:
  - [bold cyan]query (Q)[/]: Look for datasets with * wildcards (like in DAS)
  - [bold cyan]select (S)[/]: Select datasets to process further from query results
  - [bold cyan]replicas (R)[/]: Query rucio to look for files replica and then select the preferred sites
  - [bold cyan]list_selected (LS)[/]: Print a list of the selected datasets
  - [bold cyan]list_replicas (LR) index[/]: Print the selected files replicas for the selected dataset
  - [bold cyan]sites_filters[/]: show the active sites filters
  - [bold cyan]sites_filters clear[/]: clear all the active sites filters
  - [bold cyan]allowlist_sites[/]: Select sites to allowlist them for replica queries
  - [bold cyan]blocklist_sites[/]: Select sites to blocklist them for replica queries
  - [bold cyan]regex_sites[/]: Select sites with a regex for replica queries: please wrap the regex like "T[123]_(FR|IT|BE|CH|DE)_\w+"
  - [bold cyan]save (O) OUTPUTFILE[/]: Save the replicas results to file (json or yaml) for further processing
  - [bold cyan]preprocess (P) OUTPUTFILE[/]: Preprocess the replicas with dask and save the fileset to the outputfile (yaml or json)
  - [bold cyan]help[/]: get help!
"""

    if args.cli:
        console = Console()
        console.print(intro_msg, justify="left")
        app = DatasetQueryApp()
        app.cmdloop()
