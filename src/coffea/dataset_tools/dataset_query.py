from cmd2 import Cmd
import cmd2
from rich import print
from rich.pretty import pprint
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.prompt import Prompt
import rucio_utils
from collections import defaultdict
import random


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
                    f"[green bold]Y" if sel else f"[red]N",
                    end_section=ic == ncond - 1,
                )
            else:
                table.add_row(
                    "",
                    f"[bold]({j})[/bold] {c}/{'-'.join(tiers)}",
                    f"[green bold]Y" if sel else f"[red]N",
                    end_section=ic == ncond - 1,
                )
            ic += 1
            j += 1

    console.print(table)


class MyCmdApp(cmd2.Cmd):
    prompt = "\033[1;34m" + "cms-datasets" + "\033[0m > "

    def __init__(self):
        shortcuts = cmd2.DEFAULT_SHORTCUTS
        shortcuts.update(
            {
                "L": "login",
                "Q": "query",
                "R": "replicas",
                "S": "select",
                "LS": "list_selected",
                "lr": "list_results",
            }
        )
        self.console = Console()
        self.rucio_client = None
        self.selected_datasets = []
        self.last_query = ""
        self.last_query_tree = None
        self.last_query_list = None
        self.sites_whitelist = None
        self.sites_blacklist = None
        self.sites_regex = None
        self.last_replicas_results = None

        self.replica_results = defaultdict(list)
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
                args.arg_list[0], client=self.rucio_client, tree=True
            )
            # Now let's print the results as a tree
            print_dataset_query(args, outtree, self.selected_datasets, self.console)
            self.last_query = args
            self.last_query_list = outlist
            self.last_query_tree = outtree
        print("Use the command [bold red]select (S)[/] to selected the datasets")

    def do_list_results(self, args):
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
        if not self.last_query_list:
            print("First [bold red]query (Q)[/] for a dataset")
            return

        Nresults = len(self.last_query_list)
        print("[cyan]Selected datasets:")
        for s in map(int, args.arg_list):
            if s <= Nresults:
                self.selected_datasets.append(self.last_query_list[s - 1])
                print(f"- ({s}) {self.last_query_list[s-1]}")
            else:
                print(
                    f"[red]The requested dataset is not in the list. Please insert a position <={Nresults}"
                )

    def do_list_selected(self, args):
        print("[cyan]Selected datasets:")
        for i, ds in enumerate(self.selected_datasets):
            print(f"- [{i}] [blue]{ds}")

    def do_replicas(self, args):
        if len(args.arg_list) == 0:
            print(
                "[red] Please provide the index of the [bold]selected[/bold] dataset to analyze or the [bold]full dataset name[/bold]"
            )
            return

        if args.isdigit():
            if int(args) <= len(self.selected_datasets):
                dataset = self.selected_datasets[int(args) - 1]
            else:
                print(
                    f"[red]The requested dataset is not in the list. Please insert a position <={len(self.selected_datasets)}"
                )
        else:
            dataset = args.arg_list[0]

        with self.console.status(
            f"Querying rucio for replicas: [bold red]{dataset}[/]"
        ):
            outfiles, outsites, sites_counts = rucio_utils.get_dataset_files_replicas(
                dataset,
                whitelist_sites=self.sites_whitelist,
                blacklist_sites=self.sites_blacklist,
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
            table.add_row(str(i), site, f"{stat} / {Nfiles}", f"{stat*100/Nfiles:.1f}%")

        self.console.print(table)
        strategy = Prompt.ask(
            "Select sites",
            choices=["round-robin", "choice", "quit"],
            default="round-robin",
        )

        files_by_site = defaultdict(list)

        if strategy == "choice":
            ind = list(
                map(int, Prompt.ask("Enter list of sites index to be used").split(" "))
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

        # Now let's print the results
        tree = Tree(label=f"Replicas for [green]{dataset}")
        for site, files in files_by_site.items():
            T = tree.add(f"[green]{site}")
            for f in files:
                T.add(f"[cyan]{f}")

        print("Final replicas selection")
        self.console.print(tree)

    def do_whitelist_sites(self, args):
        if self.sites_whitelist == None:
            self.sites_whitelist = args.arg_list
        else:
            self.sites_whitelist += args.arg_list
        print("[green]Whitelisted sites:")
        for s in self.sites_whitelist:
            print(f"- {s}")

    def do_blacklist_sites(self, args):
        if self.sites_blacklist == None:
            self.sites_blacklist = args.arg_list
        else:
            self.sites_blacklist += args.arg_list
        print("[red]Blacklisted sites:")
        for s in self.sites_blacklist:
            print(f"- {s}")

    def do_regex_sites(self, args):
        if args.startswith('"'):
            args = args[1:]
        if args.endswith('"'):
            args = args[:-1]
        self.sites_regex = r"{}".format(args)
        print(f"New sites regex: [cyan]{self.sites_regex}")

    def do_sites_filters(self, args):
        if args == "":
            print("[green bold]Whitelisted sites:")
            if self.sites_whitelist:
                for s in self.sites_whitelist:
                    print(f"- {s}")

            print("[bold red]Blacklisted sites:")
            if self.sites_blacklist:
                for s in self.sites_blacklist:
                    print(f"- {s}")

            print(f"[bold cyan]Sites regex: [italics]{self.sites_regex}")
        if args == "clear":
            self.sites_whitelist = None
            self.sites_blacklist = None
            self.sites_regex = None
            print("[bold green]Sites filters cleared")

    def do_list_replicas(self, args):
        print("Datasets with selected replicas: ")
        for dataset in self.replica_results:
            print(f"\t-[cyan]{dataset}")


if __name__ == "__main__":
    app = MyCmdApp()
    app.cmdloop()
