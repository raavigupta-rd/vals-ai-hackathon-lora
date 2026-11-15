#include <bits/stdc++.h>
using namespace std;

static const long long NEG_INF = -(1LL << 60);
static const long long POS_INF = (1LL << 60);

struct Node {
    long long mx, mn, lm, mr, lmr, lz;
};

static inline Node merge_node(const Node& a, const Node& b) {
    Node c;
    c.mx = max(a.mx, b.mx);
    c.mn = min(a.mn, b.mn);
    c.lm = max({a.lm, b.lm, a.mx - 2 * b.mn});
    c.mr = max({a.mr, b.mr, -2 * a.mn + b.mx});
    c.lmr = max({a.lmr, b.lmr, a.lm + b.mx, a.mx + b.mr});
    c.lz = 0;
    return c;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    struct Edge {
        int u, v;
        long long w;
    };
    vector<Edge> edges(n);
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 1; i <= n - 1; i++) {
        cin >> edges[i].u >> edges[i].v >> edges[i].w;
        adj[edges[i].u].push_back({edges[i].v, i});
        adj[edges[i].v].push_back({edges[i].u, i});
    }

    // Iterative full Euler tour of length M = 2*n - 1
    int M = 2 * n - 1;
    vector<long long> init_d(M + 1, 0), dep_w(n + 1, 0);
    vector<int> first_pos(n + 1, 0), last_pos(n + 1, 0), child_end(n, 0);
    struct Frame {
        int u, p, idx;
    };
    vector<Frame> stk;
    stk.reserve(n);
    stk.push_back({1, 0, 0});
    int timer = 0;
    first_pos[1] = last_pos[1] = ++timer;
    init_d[timer] = 0;

    while (!stk.empty()) {
        auto& fr = stk.back();
        int u = fr.u;
        if (fr.idx < (int)adj[u].size()) {
            auto [v, eid] = adj[u][fr.idx++];
            if (v == fr.p) continue;
            child_end[eid] = v;
            dep_w[v] = dep_w[u] + edges[eid].w;
            first_pos[v] = last_pos[v] = ++timer;
            init_d[timer] = dep_w[v];
            stk.push_back({v, u, 0});
        } else {
            int p = fr.p;
            stk.pop_back();
            if (p != 0) {
                last_pos[p] = ++timer;
                init_d[timer] = dep_w[p];
            }
        }
    }

    vector<Node> tr(4 * M + 5);

    auto apply_add = [&](int u, long long d) {
        tr[u].mx += d;
        tr[u].mn += d;
        tr[u].lm -= d;
        tr[u].mr -= d;
        // tr[u].lmr is invariant under uniform shift +d
        tr[u].lz += d;
    };

    auto push = [&](int u) {
        if (tr[u].lz != 0) {
            apply_add(u << 1, tr[u].lz);
            apply_add(u << 1 | 1, tr[u].lz);
            tr[u].lz = 0;
        }
    };

    auto build = [&](auto& self, int u, int l, int r) -> void {
        if (l == r) {
            long long d = init_d[l];
            tr[u] = {d, d, -d, -d, 0, 0};
            return;
        }
        int m = (l + r) >> 1;
        self(self, u << 1, l, m);
        self(self, u << 1 | 1, m + 1, r);
        tr[u] = merge_node(tr[u << 1], tr[u << 1 | 1]);
    };

    auto update = [&](auto& self, int u, int l, int r, int ql, int qr, long long d) -> void {
        if (ql <= l && r <= qr) {
            apply_add(u, d);
            return;
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) self(self, u << 1, l, m, ql, qr, d);
        if (qr > m) self(self, u << 1 | 1, m + 1, r, ql, qr, d);
        long long saved_lz = 0;
        tr[u] = merge_node(tr[u << 1], tr[u << 1 | 1]);
        tr[u].lz = saved_lz;
    };

    build(build, 1, 1, M);

    while (q--) {
        int eid;
        long long nw;
        cin >> eid >> nw;
        long long delta = nw - edges[eid].w;
        edges[eid].w = nw;
        int c = child_end[eid];
        update(update, 1, 1, M, first_pos[c], last_pos[c], delta);
        cout << tr[1].lmr << '\n';
    }
    return 0;
}
