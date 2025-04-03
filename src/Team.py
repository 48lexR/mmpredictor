class Team:
    rank: int
    team: str
    wl: tuple
    net: float
    off: float
    dfs: float
    adj: float

    def __init__(self, rank: int = 0, team: str="", wl: tuple="", net: float=0.0, off: float=0.0, dfs: float=0.0, adj: float=0.0) -> None:
        self.rank = 0
        self.team = team
        self.wl = wl
        self.net = net
        self.off = off
        self.dfs = dfs
        self.adj = adj

    def asList(self) -> list:
        return [self.net, self.off, self.dfs, self.adj]