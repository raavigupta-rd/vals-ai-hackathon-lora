#include <bits/stdc++.h>
using namespace std;

struct Basis60 {
    unsigned long long b[60]{};

    void insert(unsigned long long x) {
        for (int i = 59; i >= 0; i--) {
            if (!((x >> i) & 1ULL)) continue;
            if (!b[i]) {
                b[i] = x;
                return;
            }
            x ^= b[i];
        }
    }

    unsigned long long max_xor(unsigned long long x) const {
        for (int i = 59; i >= 0; i--) {
            if ((x ^ b[i]) > x) x ^= b[i];
        }
        return x;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    if (!(cin >> n >> q)) return 0;

    vector<int> par(n + 1), sz(n + 1, 1);
    vector<unsigned long long> pot(n + 1, 0);
    vector<Basis60> basis(n + 1);
    iota(par.begin(), par.end(), 0);

    auto find_root = [&](int u) -> pair<int, unsigned long long> {
        unsigned long long xr = 0;
        while (par[u] != u) {
            xr ^= pot[u];
            u = par[u];
        }
        return {u, xr};
    };

    while (q--) {
        int tp;
        cin >> tp;
        if (tp == 1) {
            int u, v;
            unsigned long long w;
            cin >> u >> v >> w;
            auto [ru, xu] = find_root(u);
            auto [rv, xv] = find_root(v);
            if (ru == rv) {
                basis[ru].insert(xu ^ xv ^ w);
            } else {
                if (sz[ru] < sz[rv]) {
                    swap(ru, rv);
                    swap(xu, xv);
                }
                par[rv] = ru;
                pot[rv] = xu ^ xv ^ w;
                sz[ru] += sz[rv];
                for (int i = 0; i < 60; i++) {
                    if (basis[rv].b[i]) basis[ru].insert(basis[rv].b[i]);
                }
            }
        } else {
            int u, v;
            cin >> u >> v;
            auto [ru, xu] = find_root(u);
            auto [rv, xv] = find_root(v);
            if (ru != rv) {
                cout << -1 << '\n';
            } else {
                cout << basis[ru].max_xor(xu ^ xv) << '\n';
            }
        }
    }
    return 0;
}
