#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> dsu_cc(n + 1), sz_cc(n + 1, 1), bridges(n + 1, 0);
    vector<int> dsu_2ecc(n + 1), par(n + 1, 0), vis(n + 1, 0);
    iota(dsu_cc.begin(), dsu_cc.end(), 0);
    iota(dsu_2ecc.begin(), dsu_2ecc.end(), 0);
    int timer = 0;

    auto find_2ecc = [&](auto& self, int u) -> int {
        if (u == 0) return 0;
        return dsu_2ecc[u] == u ? u : dsu_2ecc[u] = self(self, dsu_2ecc[u]);
    };

    auto find_cc = [&](auto& self, int u) -> int {
        u = find_2ecc(find_2ecc, u);
        return dsu_cc[u] == u ? u : dsu_cc[u] = self(self, dsu_cc[u]);
    };

    auto make_root = [&](int u) {
        u = find_2ecc(find_2ecc, u);
        int root = u, child = 0;
        while (u != 0) {
            int p = find_2ecc(find_2ecc, par[u]);
            par[u] = child;
            dsu_cc[u] = root;
            child = u;
            u = p;
        }
        sz_cc[root] = sz_cc[child];
        bridges[root] = bridges[child];
    };

    vector<int> path_a, path_b;
    auto merge_path = [&](int a, int b) {
        ++timer;
        path_a.clear();
        path_b.clear();
        int lca = 0;
        while (lca == 0) {
            if (a != 0) {
                a = find_2ecc(find_2ecc, a);
                path_a.push_back(a);
                if (vis[a] == timer) {
                    lca = a;
                    break;
                }
                vis[a] = timer;
                a = par[a];
            }
            if (b != 0) {
                b = find_2ecc(find_2ecc, b);
                path_b.push_back(b);
                if (vis[b] == timer) {
                    lca = b;
                    break;
                }
                vis[b] = timer;
                b = par[b];
            }
        }
        int rc = find_cc(find_cc, lca);
        for (int v : path_a) {
            dsu_2ecc[v] = lca;
            if (v == lca) break;
            bridges[rc]--;
        }
        for (int v : path_b) {
            dsu_2ecc[v] = lca;
            if (v == lca) break;
            bridges[rc]--;
        }
    };

    auto add_edge = [&](int a, int b) {
        a = find_2ecc(find_2ecc, a);
        b = find_2ecc(find_2ecc, b);
        if (a == b) return;
        int ca = find_cc(find_cc, a);
        int cb = find_cc(find_cc, b);
        if (ca != cb) {
            if (sz_cc[ca] > sz_cc[cb]) {
                swap(a, b);
                swap(ca, cb);
            }
            make_root(a);
            par[a] = b;
            dsu_cc[a] = b;
            sz_cc[cb] += sz_cc[a];
            bridges[cb] += bridges[a] + 1;
        } else {
            merge_path(a, b);
        }
    };

    long long last = 0;
    while (q--) {
        int tp, rx, ry;
        cin >> tp >> rx >> ry;
        int u = (int)((rx + last) % n) + 1;
        int v = (int)((ry + last) % n) + 1;
        if (tp == 1) {
            add_edge(u, v);
        } else {
            int cu = find_cc(find_cc, u);
            int cv = find_cc(find_cc, v);
            if (cu != cv) {
                cout << "-1 0\n";
            } else {
                int same2 = (find_2ecc(find_2ecc, u) == find_2ecc(find_2ecc, v)) ? 1 : 0;
                int br = bridges[cu];
                cout << same2 << ' ' << br << '\n';
                last = (last + same2 + br) % n;
            }
        }
    }
    return 0;
}
