#include <bits/stdc++.h>
using namespace std;

static const long long INF = (1LL << 60);

struct Dinic {
    struct Edge {
        int to, rev;
        long long cap;
    };
    int n;
    vector<vector<Edge>> g;
    vector<int> lvl, ptr;

    explicit Dinic(int n_) : n(n_), g(n_), lvl(n_), ptr(n_) {}

    void add_edge(int u, int v, long long cap) {
        Edge a{v, (int)g[v].size(), cap};
        Edge b{u, (int)g[u].size(), 0};
        g[u].push_back(a);
        g[v].push_back(b);
    }

    bool bfs(int s, int t) {
        fill(lvl.begin(), lvl.end(), -1);
        queue<int> q;
        lvl[s] = 0;
        q.push(s);
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            for (const auto& e : g[u]) {
                if (e.cap > 0 && lvl[e.to] == -1) {
                    lvl[e.to] = lvl[u] + 1;
                    q.push(e.to);
                }
            }
        }
        return lvl[t] != -1;
    }

    long long dfs(int u, int t, long long pushed) {
        if (pushed == 0 || u == t) return pushed;
        for (int& cid = ptr[u]; cid < (int)g[u].size(); cid++) {
            auto& e = g[u][cid];
            if (e.cap > 0 && lvl[e.to] == lvl[u] + 1) {
                long long tr = dfs(e.to, t, min(pushed, e.cap));
                if (tr > 0) {
                    e.cap -= tr;
                    g[e.to][e.rev].cap += tr;
                    return tr;
                }
            }
        }
        return 0;
    }

    long long max_flow(int s, int t) {
        long long flow = 0;
        while (bfs(s, t)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (long long tr = dfs(s, t, INF)) flow += tr;
        }
        return flow;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    if (!(cin >> n >> m)) return 0;

    vector<vector<tuple<int, long long, int>>> adj(n + 1);
    for (int i = 1; i <= n - 1; i++) {
        int u, v;
        long long c;
        cin >> u >> v >> c;
        adj[u].push_back({v, c, i});
        adj[v].push_back({u, c, i});
    }

    vector<int> par(n + 1, 0), dep(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0);
    vector<long long> up_cost(n + 1, 0);
    vector<int> ord;
    ord.reserve(n);
    ord.push_back(1);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (auto [v, c, id] : adj[u]) {
            if (v != par[u]) {
                par[v] = u;
                dep[v] = dep[u] + 1;
                up_cost[v] = c;
                ord.push_back(v);
            }
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        int best = 0;
        for (auto [v, c, id] : adj[u]) {
            if (v != par[u]) {
                sz[u] += sz[v];
                if (sz[v] > best) {
                    best = sz[v];
                    heavy[u] = v;
                }
            }
        }
    }

    vector<int> head(n + 1, 0), tin(n + 1, 0), inv_tin(n + 1, 0);
    int timer = 0;
    vector<pair<int, int>> stk;
    stk.push_back({1, 1});
    while (!stk.empty()) {
        auto [u, h] = stk.back();
        stk.pop_back();
        for (int x = u; x != 0; x = heavy[x]) {
            head[x] = h;
            tin[x] = ++timer;
            inv_tin[timer] = x;
            for (auto [v, c, id] : adj[x]) {
                if (v != par[x] && v != heavy[x]) {
                    stk.push_back({v, v});
                }
            }
        }
    }

    // Nodes in flow network:
    // 0: source, 1: sink
    // [2 .. 2 + 4*n]: segtree over HLD positions [2..n] (tin[1]=1 has no parent edge)
    // [2 + 4*n + 1 .. 2 + 4*n + m]: courier nodes
    int seg_base = 2;
    int cour_base = seg_base + 4 * n + 2;
    Dinic dinic(cour_base + m + 2);
    int S = 0, T = 1;

    if (n >= 2) {
        auto build_seg = [&](auto& self, int u, int l, int r) -> void {
            if (l == r) {
                int vtx = inv_tin[l];
                dinic.add_edge(seg_base + u, T, up_cost[vtx]);
                return;
            }
            int mid = (l + r) >> 1;
            self(self, u << 1, l, mid);
            self(self, u << 1 | 1, mid + 1, r);
            dinic.add_edge(seg_base + u, seg_base + (u << 1), INF);
            dinic.add_edge(seg_base + u, seg_base + (u << 1 | 1), INF);
        };
        build_seg(build_seg, 1, 2, n);
    }

    auto add_range = [&](auto& self, int cnode, int u, int l, int r, int ql, int qr) -> void {
        if (ql > qr) return;
        if (ql <= l && r <= qr) {
            dinic.add_edge(cnode, seg_base + u, INF);
            return;
        }
        int mid = (l + r) >> 1;
        if (ql <= mid) self(self, cnode, u << 1, l, mid, ql, qr);
        if (qr > mid) self(self, cnode, u << 1 | 1, mid + 1, r, ql, qr);
    };

    for (int j = 0; j < m; j++) {
        int u, v;
        long long p;
        cin >> u >> v >> p;
        if (u == v) continue;
        int cnode = cour_base + j;
        dinic.add_edge(S, cnode, p);
        while (head[u] != head[v]) {
            if (dep[head[u]] < dep[head[v]]) swap(u, v);
            add_range(add_range, cnode, 1, 2, n, max(2, tin[head[u]]), tin[u]);
            u = par[head[u]];
        }
        if (u != v) {
            if (dep[u] > dep[v]) swap(u, v);
            add_range(add_range, cnode, 1, 2, n, tin[u] + 1, tin[v]);
        }
    }

    cout << dinic.max_flow(S, T) << '\n';
    return 0;
}
