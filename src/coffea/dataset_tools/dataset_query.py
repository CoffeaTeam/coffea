import gzip
import json
import os
import random
from collections import defaultdict
from typing import List

import yaml
from dask.distributed import Client
from rich import print
from rich.console import Console
from rich.prompt import Confirm, IntPrompt, Prompt
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


def get_indices_query(input_str: str, maxN: int) -> List[int]:
    tokens = input_str.strip().split(" ")
    final_tokens = []
    for t in tokens:
        if t.isdigit():
            if int(t) > maxN:
                print(
                    f"[red bold]Requested index {t} larger than available elements {maxN}"
                )
                return False
            final_tokens.append(int(t) - 1)  # index 0
        elif "-" in t:
            rng = t.split("-")
            try:
                for i in range(
                    int(rng[0]), int(rng[1]) + 1
                ):  # including the last index
                    if i > maxN:
                        print(
                            f"[red bold]Requested index {t} larger than available elements {maxN}"
                        )
                        return False
                    final_tokens.append(i - 1)
            except Exception:
                print(
                    "[red]Error! Bad formatting for selection string. Use e.g. 1 4 5-9"
                )
                return False
        elif t == "all":
            final_tokens = list(range(0, maxN))
        else:
            print("[red]Error! Bad formatting for selection string. Use e.g. 1 4 5-9")
            return False
    return final_tokens


class DataDiscoveryCLI:
    def __init__(self):
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

        self.commands = [
            "help",
            "login",
            "query",
            "query-results",
            "select",
            "list-selected",
            "replicas",
            "list-replicas",
            "save",
            "preprocess",
            "allow-sites",
            "block-sites",
            "regex-sites",
            "sites-filters",
            "quit",
        ]

    def start_cli(self):
        while True:
            command = Prompt.ask(">", choices=self.commands)
            if command == "help":
                print(
                    r"""[bold yellow]Welcome to the datasets discovery coffea CLI![/bold yellow]
Use this CLI tool to query the CMS datasets and to select interactively the grid sites to use for reading the files in your analysis.
Some basic commands:
  - [bold cyan]query[/]: Look for datasets with * wildcards (like in DAS)
  - [bold cyan]select[/]: Select datasets to process further from query results
  - [bold cyan]replicas[/]: Query rucio to look for files replica and then select the preferred sites
  - [bold cyan]query-results[/]: List the results of the last dataset query
  - [bold cyan]list-selected[/]: Print a list of the selected datasets
  - [bold cyan]list-replicas[/]: Print the selected files replicas for the selected dataset
  - [bold cyan]sites-filters[/]: show the active sites filters and aks to clear them
  - [bold cyan]allow-sites[/]: Restrict the grid sites available for replicas query only to the requested list
  - [bold cyan]block-sites[/]: Exclude grid sites from the available sites for replicas query
  - [bold cyan]regex-sites[/]: Select sites with a regex for replica queries: e.g.  "T[123]_(FR|IT|BE|CH|DE)_\w+"
  - [bold cyan]save[/]: Save the replicas query results to file (json or yaml) for further processing
  - [bold cyan]preprocess[/]: Preprocess the replicas with dask and save the fileset for further processing with uproot/coffea
  - [bold cyan]help[/]: Print this help message
            """
                )
            elif command == "login":
                self.do_login()
            elif command == "quit":
                print("Bye!")
                break
            elif command == "query":
                self.do_query()
            elif command == "query-results":
                self.do_query_results()
            elif command == "select":
                self.do_select()
            elif command == "list-selected":
                self.do_list_selected()
            elif command == "replicas":
                self.do_replicas()
            elif command == "list-replicas":
                self.do_list_replicas()
            elif command == "save":
                self.do_save()
            elif command == "preprocess":
                self.do_preprocess()
            elif command == "allow-sites":
                self.do_allowlist_sites()
            elif command == "block-sites":
                self.do_blocklist_sites()
            elif command == "regex-sites":
                self.do_regex_sites()
            elif command == "sites-filters":
                self.do_sites_filters()
            else:
                break

    def do_login(self, proxy=None):
        """Login to the rucio client. Optionally a specific proxy file can be passed to the command.
        If the proxy file is not specified, `voms-proxy-info` is used"""
        if proxy:
            self.rucio_client = rucio_utils.get_rucio_client(proxy)
        else:
            self.rucio_client = rucio_utils.get_rucio_client()
        print(self.rucio_client)

    def do_whoami(self):
        # Your code here
        if not self.rucio_client:
            print("First [bold]login (L)[/] to the rucio server")
            return
        print(self.rucio_client.whoami())

    def do_query(self):
        # Your code here
        query = Prompt.ask(
            "[yellow bold]Query for[/]",
        )
        with self.console.status(f"Querying rucio for: [bold red]{query}[/]"):
            outlist, outtree = rucio_utils.query_dataset(
                query,
                client=self.rucio_client,
                tree=True,
                scope="cms",  # TODO configure scope
            )
            # Now let's print the results as a tree
            print_dataset_query(query, outtree, self.selected_datasets, self.console)
            self.last_query = query
            self.last_query_list = outlist
            self.last_query_tree = outtree
        print("Use the command [bold red]select (S)[/] to selected the datasets")

    def do_query_results(self):
        if self.last_query_list:
            print_dataset_query(
                self.last_query,
                self.last_query_tree,
                self.selected_datasets,
                self.console,
            )
        else:
            print("First [bold red]query (Q)[/] for a dataset")

    def do_select(self):
        """Selected the datasets from the list of query results. Input a list of indices
        also with range 4-6 or "all"."""
        if not self.last_query_list:
            print("First [bold red]query (Q)[/] for a dataset")
            return

        selection = Prompt.ask(
            "[yellow bold]Select datasets indices[/] (e.g 1 4 6-10)", default="all"
        )
        final_tokens = get_indices_query(selection, len(self.last_query_list))
        if not final_tokens:
            return

        Nresults = len(self.last_query_list)
        print("[cyan]Selected datasets:")

        for s in final_tokens:
            if s < Nresults:
                self.selected_datasets.append(self.last_query_list[s])
                print(f"- ({s+1}) {self.last_query_list[s]}")
            else:
                print(
                    f"[red]The requested dataset is not in the list. Please insert a position <={Nresults}"
                )

    def do_list_selected(self):
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

    def do_replicas(self):
        selection = Prompt.ask(
            "[yellow bold]Select datasets indices[/] (e.g 1 4 6-10)", default="all"
        )
        indices = get_indices_query(selection, len(self.selected_datasets))
        if not indices:
            return
        datasets = [self.selected_datasets[ind] for ind in indices]

        for dataset in datasets:
            with self.console.status(
                f"Querying rucio for replicas: [bold red]{dataset}[/]"
            ):
                try:
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
                except Exception as e:
                    print(f"\n[red bold] Exception: {e}[/]")
                    return
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

    def do_allowlist_sites(self):
        sites = Prompt.ask(
            "[yellow]Restrict the available sites to (comma-separated list)"
        ).split(",")
        if self.sites_allowlist is None:
            self.sites_allowlist = sites
        else:
            self.sites_allowlist += sites
        print("[green]Allowlisted sites:")
        for s in self.sites_allowlist:
            print(f"- {s}")

    def do_blocklist_sites(self):
        sites = Prompt.ask("[yellow]Exclude the sites (comma-separated list)").split(
            ","
        )
        if self.sites_blocklist is None:
            self.sites_blocklist = sites
        else:
            self.sites_blocklist += sites
        print("[red]Blocklisted sites:")
        for s in self.sites_blocklist:
            print(f"- {s}")

    def do_regex_sites(self):
        regex = Prompt.ask("[yellow]Regex to restrict the available sites")
        if len(regex):
            self.sites_regex = rf"{regex}"
            print(f"New sites regex: [cyan]{self.sites_regex}")

    def do_sites_filters(self):
        print("[green bold]Allow-listed sites:")
        if self.sites_allowlist:
            for s in self.sites_allowlist:
                print(f"- {s}")

        print("[bold red]Block-listed sites:")
        if self.sites_blocklist:
            for s in self.sites_blocklist:
                print(f"- {s}")

        print(f"[bold cyan]Sites regex: [italics]{self.sites_regex}")

        if Confirm.ask("Clear sites restrinction?", default=False):
            self.sites_allowlist = None
            self.sites_blocklist = None
            self.sites_regex = None
            print("[bold green]Sites filters cleared")

    def do_list_replicas(self):
        selection = Prompt.ask(
            "[yellow bold]Select datasets indices[/] (e.g 1 4 6-10)", default="all"
        )
        indices = get_indices_query(selection, len(self.selected_datasets))
        datasets = [self.selected_datasets[ind] for ind in indices]

        for dataset in datasets:
            if dataset not in self.replica_results:
                print(
                    f"[red bold]No replica info for dataset {dataset}. You need to selected the replicas with [cyan] replicas [/cyan] command[/]"
                )
                return
            tree = Tree(label=f"[bold orange]Replicas for [/][green]{dataset}[/]")
            for site, files in self.replica_results_bysite[dataset].items():
                T = tree.add(f"[green]{site}")
                for f in files:
                    T.add(f"[cyan]{f}")

            self.console.print(tree)

    def do_save(self, filename=None):
        """Save the replica information in yaml format"""
        if not filename:
            filename = Prompt.ask(
                "[yellow bold]Output file name (.yaml or .json)", default="output.json"
            )
        format = os.path.splitext(filename)[1]
        output = {}
        for fileset, files in self.replica_results.items():
            output[fileset] = {"files": files, "metadata": {}}
        with open(filename, "w") as file:
            if format == ".yaml":
                yaml.dump(output, file, default_flow_style=False)
            elif format == ".json":
                json.dump(output, file, indent=2)
        print(f"[green]File {filename} saved!")

    def do_preprocess(
        self,
        output_file=None,
        step_size=None,
        align_to_clusters=None,
        dask_cluster=None,
    ):
        """Perform preprocessing for concrete fileset extraction.
        Args:  output_file [step_size] [align to file cluster boundaries] [dask cluster url]
        """
        if not output_file:
            output_file = Prompt.ask(
                "[yellow bold]Output name", default="output_preprocessing"
            )
        if not step_size:
            step_size = IntPrompt.ask("[yellow bold]Step size", default=None)
        if align_to_clusters is None:
            align_to_clusters = Confirm.ask(
                "[yellow bold]Align to clusters", default=True
            )
        if not dask_cluster:
            dask_cluster = Prompt.ask("[yellow bold]Dask cluster url", default="None")
            if dask_cluster == "None":
                dask_cluster = None

        replicas = {}
        for fileset, files in self.replica_results.items():
            replicas[fileset] = {"files": {f: "Events" for f in files}, "metadata": {}}
        # init a local Dask cluster
        with self.console.status(
            "[red] Preprocessing files to extract available chunks with dask[/]"
        ):
            with Client(dask_cluster) as _:
                out_available, out_updated = preprocess(
                    replicas,
                    maybe_step_size=step_size,
                    align_clusters=align_to_clusters,
                    skip_bad_files=True,
                )
        with gzip.open(f"{output_file}_available.json.gz", "wt") as file:
            print(f"Saved available fileset chunks to {output_file}_available.json.gz")
            json.dump(out_available, file, indent=2)
        with gzip.open(f"{output_file}_all.json.gz", "wt") as file:
            print(f"Saved all fileset chunks to {output_file}_all.json.gz")
            json.dump(out_updated, file, indent=2)


if __name__ == "__main__":
    cli = DataDiscoveryCLI()
    cli.start_cli()
