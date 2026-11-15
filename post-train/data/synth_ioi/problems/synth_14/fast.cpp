#include <bits/stdc++.h>
using namespace std;

struct RollbackParityDSU {
    vector<int> p, sz, xr;
    vector<pair<int, int>> hist; // (child, old_sz_parent)

    RollbackParityDSU(int n) : p(n + 1), sz(n + 1, 1), xr(n + 1, 0) {
        iota(p.begin(), p.end(), 0);
    }
    pair<int, int> find(int x) const {
        int par = 0;
        while (p[x] != x) {
            par ^= xr[x];
            x = p[x];
        }
        return {x, par};
    }
    bool add_edge(int u, int v) {
        auto [ru, pu] = find(u);
        auto [rv, pv] = find(v);
        if (ru == rv) return (pu ^ pv) == 1;
        if (sz[ru] < sz[rv]) { swap(ru, rv); swap(pu, pv); }
        hist.push_back({rv, sz[ru]});
        p[rv] = ru;
        xr[rv] = pu ^ pv ^ 1;
        sz[ru] += sz[rv];
        return true;
    }
    void rollback(size_t snap) {
        while (hist.size() > snap) {
            auto [rv, old_sz] = hist.back();
            hist.pop_back();
            int ru = p[rv];
            sz[ru] = old_sz;
            p[rv] = rv;
            xr[rv] = 0;
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, T;
    cin >> n >> m >> T;
    vector<vector<pair<int, int>>> seg(4 * T + 4);

    function<void(int, int, int, int, int, pair<int, int>)> insert_edge =
        [&](int u, int l, int r, int ql, int qr, pair<int, int> e) {
            if (ql <= l && r <= qr) {
                seg[u].push_back(e);
                return;
            }
            int mid = (l + r) >> 1;
            if (ql <= mid) insert_edge(2 * u, l, mid, ql, qr, e);
            if (qr > mid) insert_edge(2 * u + 1, mid + 1, r, ql, qr, e);
        };

    for (int i = 0; i < m; i++) {
        int u, v, l, r;
        cin >> u >> v >> l >> r;
        insert_edge(1, 1, T, l, r, {u, v});
    }

    RollbackParityDSU dsu(n);
    vector<char> ans(T + 1, 0);

    function<void(int, int, int, bool)> solve = [&](int u, int l, int r, bool ok) {
        size_t snap = dsu.hist.size();
        if (ok) {
            for (auto [a, b] : seg[u]) {
                if (!dsu.add_edge(a, b)) { ok = false; break; }
            }
        }
        if (l == r) {
            ans[l] = ok ? 1 : 0;
        } else {
            int mid = (l + r) >> 1;
            solve(2 * u, l, mid, ok);
            solve(2 * u + 1, mid + 1, r, ok);
        }
        dsu.rollback(snap);
    };

    solve(1, 1, T, true);
    for (int t = 1; t <= T; t++) cout << (ans[t] ? "YES\n" : "NO\n");
    return 0;
}
