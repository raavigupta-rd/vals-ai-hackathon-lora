#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<pair<int, int>> es(n - 1);
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        cin >> es[i].first >> es[i].second;
        adj[es[i].first].push_back(es[i].second);
        adj[es[i].second].push_back(es[i].first);
    }

    vector<int> dep(n + 1, 0), par(n + 1, 0), first_pos(n + 1, 0), euler;
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
                int v = adj[u][idx++];
                if (v == par[u]) continue;
                par[v] = u;
                dep[v] = dep[u] + 1;
                st.push_back({v, 0});
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
    auto dist = [&](int u, int v) {
        int l = first_pos[u], r = first_pos[v];
        if (l > r) swap(l, r);
        int k = __lg(r - l + 1);
        int a = rmq[k][l], b = rmq[k][r - (1 << k) + 1];
        int w = dep[a] < dep[b] ? a : b;
        return dep[u] + dep[v] - 2 * dep[w];
    };

    vector<int> p(n + 1), sz(n + 1, 1), e1(n + 1), e2(n + 1), diam(n + 1, 0);
    iota(p.begin(), p.end(), 0);
    iota(e1.begin(), e1.end(), 0);
    iota(e2.begin(), e2.end(), 0);
    function<int(int)> findp = [&](int x) { return p[x] == x ? x : p[x] = findp(p[x]); };

    for (auto [u, v] : es) {
        int a = findp(u), b = findp(v);
        if (sz[a] < sz[b]) swap(a, b);
        int cand[4] = {e1[a], e2[a], e1[b], e2[b]};
        int best_d = -1, ba = cand[0], bb = cand[0];
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                int d = dist(cand[i], cand[j]);
                if (d > best_d) { best_d = d; ba = cand[i]; bb = cand[j]; }
            }
        }
        p[b] = a;
        sz[a] += sz[b];
        e1[a] = ba; e2[a] = bb; diam[a] = best_d;
        cout << best_d << '\n';
    }
    return 0;
}
