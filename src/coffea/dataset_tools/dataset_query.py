from cmd2 import Cmd
import cmd2
from rich import print
from rich.pretty import pprint
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
import rucio_utils


def print_dataset_query(query, dataset_list, selected, console):
    table = Table(title=f"Query: [bold red]{query}")
    table.add_column("name", justify="left", style="cyan", no_wrap=True)
    table.add_column("tag", style="magenta", no_wrap=True)
    table.add_column("selected", justify="center")
    table.row_styles = ["dim", "none"]
    j = 1
    for name, conds in dataset_list.items():
        ic = 0
        ncond = len(conds)
        for c, tiers in conds.items():
            dataset = f"/{name}/{c}/{tiers[0]}"
            sel = dataset in selected
            if ic ==0:
                table.add_row(name, f"[bold]({j})[/bold] {c}/{'-'.join(tiers)}",
                              f"[green]Y" if sel else f"[red]N",
                              end_section = ic==ncond-1)
            else:
                table.add_row("",  f"[bold]({j})[/bold] {c}/{'-'.join(tiers)}",
                              f"[green]Y" if sel else f"[red]N",
                              end_section = ic==ncond-1)
            ic+=1
            j+=1

    console.print(table)


class MyCmdApp(cmd2.Cmd):

    prompt = "\033[1;34m" + "cms-datasets" + "\033[0m > "
    
    def __init__(self):
        shortcuts = cmd2.DEFAULT_SHORTCUTS
        shortcuts.update({ 'L': 'login', 'Q': 'query', 'R': 'replicas',
                           'S': 'select',
                           'lr': 'list_results'})
        self.console = Console()
        self.rucio_client = None
        self.selected_datasets = [ ]
        self.last_query = ""
        self.last_query_results = None
        super().__init__(shortcuts=shortcuts)

    def do_login(self, args):
        '''Login to the rucio client. Optionally a specific proxy file can be passed to the command.
        If the proxy file is not specified, `voms-proxy-info` is used'''
        if args:
            self.rucio_client = rucio_utils.get_rucio_client(args[0])
        else:
            self.rucio_client = rucio_utils.get_rucio_client()
            
        print(self.rucio_client)
        #pprint(self.rucio_client.whoami())

    def do_whoami(self, args):
        # Your code here
        if not self.rucio_client:
            print("First [bold]login (L)[/] to the rucio server")
            return
        print(self.rucio_client.whoami())
        
    def do_query(self, args):
        # Your code here
        with self.console.status(f"Querying rucio for: [bold red]{args}[/]"):    
            out = rucio_utils.query_dataset(args.arg_list[0],
                                            client=self.rucio_client,
                                            tree=True)
            # Now let's print the results as a tree
            print_dataset_query(args, out,
                               self.selected_datasets,
                               self.console)
            self.last_query = args
            self.last_query_results = out

    def do_list_results(self, args):
        if self.last_query_results:
            print_dataset_query(self.last_query, self.last_query_results,
                            self.selected_datasets, self.console)
        else:
            print("First [bold red]query (Q)[/] for a dataset")

    def do_select(self, args):
        if not self.last_query_results:
            print("First [bold red]query (Q)[/] for a dataset")
            return

        for s in map(int, args.arg_list):
            print(s)
        
    
        
        
    def do_replicas(self, args):
        # Your code here
        self.poutput("Replicas command executed")

if __name__ == "__main__":
    app = MyCmdApp()
    app.cmdloop()
