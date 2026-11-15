#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> p, sz;
    DSU(int n) : p(n), sz(n, 1) { iota(p.begin(), p.end(), 0); }
    int find(int x) { return p[x] == x ? x : p[x] = find(p[x]); }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (sz[a] < sz[b]) swap(a, b);
        p[b] = a; sz[a] += sz[b];
        return true;
    }
};

struct Edge {
    long long w;
    int u, v;
    bool operator<(const Edge& o) const { return w < o.w; }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> ox(n), oy(n);
    for (int i = 0; i < n; i++) cin >> ox[i] >> oy[i];
    if (n <= 1) {
        cout << 0 << '\n';
        return 0;
    }

    const long long INF = (1LL << 61);
    vector<Edge> edges;
    edges.reserve(8 * n);

    vector<long long> tx(n), ty(n), diffs(n);
    vector<int> ord(n);
    vector<pair<long long, int>> bit;

    for (int fx : {1, -1}) {
        for (int fy : {1, -1}) {
            for (int sw : {0, 1}) {
                diffs.resize(n);
                for (int i = 0; i < n; i++) {
                    long long x = fx * ox[i], y = fy * oy[i];
                    if (sw) swap(x, y);
                    tx[i] = x;
                    ty[i] = y;
                    diffs[i] = x - y;
                }
                sort(diffs.begin(), diffs.end());
                diffs.erase(unique(diffs.begin(), diffs.end()), diffs.end());
                int m = (int)diffs.size();

                iota(ord.begin(), ord.end(), 0);
                sort(ord.begin(), ord.end(), [&](int a, int b) {
                    if (ty[a] != ty[b]) return ty[a] > ty[b];
                    if (tx[a] != tx[b]) return tx[a] > tx[b];
                    return a > b;
                });

                bit.assign(m + 1, {INF, -1});
                auto upd = [&](int idx, pair<long long, int> val) {
                    for (; idx <= m; idx += idx & -idx)
                        if (val < bit[idx]) bit[idx] = val;
                };
                auto qry = [&](int idx) {
                    pair<long long, int> res = {INF, -1};
                    for (; idx > 0; idx -= idx & -idx)
                        if (bit[idx] < res) res = bit[idx];
                    return res;
                };

                for (int u : ord) {
                    int rk = (int)(lower_bound(diffs.begin(), diffs.end(), tx[u] - ty[u]) - diffs.begin());
                    int pos = m - rk; // reversed so prefix [1..pos] means tx - ty >= tx[u] - ty[u]
                    auto [_, v] = qry(pos);
                    if (v != -1) {
                        long long w = llabs(ox[u] - ox[v]) + llabs(oy[u] - oy[v]);
                        edges.push_back({w, u, v});
                    }
                    upd(pos, {tx[u] + ty[u], u});
                }
            }
        }
    }

    sort(edges.begin(), edges.end());
    DSU dsu(n);
    long long total = 0;
    int taken = 0;
    for (const auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            total += e.w;
            if (++taken == n - 1) break;
        }
    }
    cout << total << '\n';
    return 0;
}
