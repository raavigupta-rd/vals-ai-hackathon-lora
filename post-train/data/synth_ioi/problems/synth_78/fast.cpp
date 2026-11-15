#include <bits/stdc++.h>
using namespace std;

struct AdjEdge { int to, col; long long w; };
struct Req { int qid, col, coeff; long long nw; };

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<vector<AdjEdge>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v, c; long long w;
        cin >> u >> v >> c >> w;
        adj[u].push_back({v, c, w});
        adj[v].push_back({u, c, w});
    }

    // Iterative Euler tour for O(1) RMQ LCA + base root distances
    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
    vector<long long> rdist(n + 1, 0);
    euler.reserve(2 * n);
    {
        vector<pair<int, int>> st = {{1, 0}};
        while (!st.empty()) {
            auto& [u, idx] = st.back();
            if (idx == 0) {
                first_pos[u] = (int)euler.size();
                euler.push_back(u);
            }
            if (idx < (int)adj[u].size()) {
                auto e = adj[u][idx++];
                if (e.to == par[u]) continue;
                par[e.to] = u;
                dep[e.to] = dep[u] + 1;
                rdist[e.to] = rdist[u] + e.w;
                st.push_back({e.to, 0});
            } else {
                st.pop_back();
                if (!st.empty()) euler.push_back(st.back().first);
            }
        }
    }
    int m = (int)euler.size(), LOG = __lg(m) + 1;
    vector<vector<int>> rmq(LOG, vector<int>(m));
    rmq[0] = euler;
    for (int k = 1; k < LOG; k++) {
        int h = 1 << (k - 1);
        for (int i = 0; i + (1 << k) <= m; i++) {
            int a = rmq[k - 1][i], b = rmq[k - 1][i + h];
            rmq[k][i] = dep[a] < dep[b] ? a : b;
        }
    }
    auto lca = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        return dep[a] < dep[b] ? a : b;
    };

    vector<vector<Req>> reqs(n + 1);
    for (int i = 0; i < q; i++) {
        int a, b, col; long long nw;
        cin >> a >> b >> col >> nw;
        int w = lca(a, b);
        reqs[a].push_back({i, col, +1, nw});
        reqs[b].push_back({i, col, +1, nw});
        reqs[w].push_back({i, col, -2, nw});
    }

    vector<long long> ans(q, 0), sum_col(n + 1, 0);
    vector<int> cnt_col(n + 1, 0);

    struct Frame { int u, idx, in_col; long long in_w; };
    vector<Frame> st = {{1, 0, 0, 0}};
    while (!st.empty()) {
        auto& f = st.back();
        int u = f.u;
        if (f.idx == 0) {
            if (f.in_col) {
                cnt_col[f.in_col]++;
                sum_col[f.in_col] += f.in_w;
            }
            for (const auto& rq : reqs[u]) {
                long long cur = rdist[u] - sum_col[rq.col] + rq.nw * cnt_col[rq.col];
                ans[rq.qid] += rq.coeff * cur;
            }
        }
        bool pushed = false;
        while (f.idx < (int)adj[u].size()) {
            auto e = adj[u][f.idx++];
            if (e.to == par[u]) continue;
            st.push_back({e.to, 0, e.col, e.w});
            pushed = true;
            break;
        }
        if (!pushed) {
            if (f.in_col) {
                cnt_col[f.in_col]--;
                sum_col[f.in_col] -= f.in_w;
            }
            st.pop_back();
        }
    }

    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
